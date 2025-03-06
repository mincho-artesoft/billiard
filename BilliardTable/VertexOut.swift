import SwiftUI
import MetalKit
import simd

// MARK: - Metal Shader Source
let metalShader = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Ball {
    float2 position;
    float2 velocity;
    float4 quaternion;
};

// ================== Utility Functions ===================
float3 hsvToRgb(float3 c) {
    float3 p = abs(fract(c.xxx + float3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(float3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

float prBoxDf(float3 p, float3 b) {
    float3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

float prRoundBoxDf(float3 p, float3 b, float r) {
    return length(max(abs(p) - b, 0.0)) - r;
}

float prSphDf(float3 p, float r) {
    return length(p) - r;
}

float prRoundCylDf(float3 p, float r, float rt, float h) {
    float dxy = length(p.xy) - (r - (rt / 2.5) * p.z);
    float dz = abs(p.z) - h;
    return min(min(max(dxy + rt, dz), max(dxy, dz + rt)), length(float2(dxy, dz) + rt) - rt);
}

float2 rot2D(float2 q, float a) {
    float2 cs = float2(cos(a), sin(a));
    return float2(q.x * cs.x - q.y * cs.y, q.x * cs.y + q.y * cs.x);
}

float smoothMin(float a, float b, float r) {
    float h = clamp(0.5 + 0.5 * (b - a) / r, 0.0, 1.0);
    return mix(b, a, h) - r * h * (1.0 - h);
}

float smoothMax(float a, float b, float r) {
    return -smoothMin(-a, -b, r);
}

// -------------------- Rotations in 3D --------------------
float3 rotateX(float3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3(v.x, c*v.y - s*v.z, s*v.y + c*v.z);
}

float3 rotateY(float3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3(c*v.x + s*v.z, v.y, -s*v.x + c*v.z);
}

float3x3 qtToRMat(float4 q) {
    q = normalize(q);
    float s = q.w * q.w - 0.5;
    float3x3 m;
    m[0][0] = q.x * q.x + s; m[1][1] = q.y * q.y + s; m[2][2] = q.z * q.z + s;
    float a1 = q.x * q.y; float a2 = q.z * q.w; m[0][1] = a1 + a2; m[1][0] = a1 - a2;
    a1 = q.x * q.z; a2 = q.y * q.w; m[2][0] = a1 + a2; m[0][2] = a1 - a2;
    a1 = q.y * q.z; a2 = q.x * q.w; m[1][2] = a1 + a2; m[2][1] = a1 - a2;
    return 2.0 * m;
}

// ================ Ball Intersection Helper ================
void ballHit(float3 ro, float3 rd, thread float &dist, thread float3 &normal,
             thread int &id, constant Ball* balls [[buffer(2)]]) {
    const int nBall = 16;
    const float rad = 0.47;
    dist = 50.0;
    normal = float3(0.0);
    id = -1;
    for (int n = 0; n < nBall; n++) {
        float3 u = ro - float3(balls[n].position.x, 0.05, balls[n].position.y);
        float b = dot(rd, u);
        float w = b * b - dot(u, u) + rad * rad;
        if (w > 0.0) {
            float d = -b - sqrt(w);
            if (d > 0.0 && d < dist) {
                dist = d;
                normal = normalize(u + d * rd);
                id = n;
            }
        }
    }
}

// ================ Scene Rendering ================
float3 showScene(float3 ro, float3 rd,
                 float time,
                 float cueOffset,
                 float2 cueTipOffset,
                 constant Ball* balls [[buffer(2)]],
                 int cueVisible,
                 // 2D: original single angle
                 float cueAngle,
                 // 3D: new yaw & pitch
                 float2 cue3DRotate)
{
    // Table, geometry constants
    const float hbLen = 8.0;
    const float bWid  = 0.4;
    const float2 hIn  = float2(hbLen, hbLen * 1.75) - bWid;
    const float PI    = 3.14159;

    float3 col = float3(0.05, 0.05, 0.1);
    float t = 0.0;
    const float maxDist = 50.0;

    // Ray vs. White Ball (and others)
    float dstBall;
    float3 ballNormal;
    int ballId;
    ballHit(ro, rd, dstBall, ballNormal, ballId, balls);

    float dstTable = maxDist;
    float dstCue   = maxDist;
    float3 cueHitPos;

    // Marching until hitting table, or the cue, or the ball
    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * t;
        float dSurface = prBoxDf(p, float3(hIn.x, 0.4, hIn.y));
        float3 pb = p; pb.y -= -0.6;
        float dBorder = prRoundBoxDf(pb, float3(hIn.x + 0.6, 0.5, hIn.y + 0.6), 0.2);

        float dTable = max(dBorder, -dSurface);

        float2 hInPocket = hIn - bWid + 0.03;
        float3 q = p;
        q.x = abs(q.x) - hInPocket.x;
        q.z = fmod(q.z + 0.5 * hInPocket.y, hInPocket.y) - 0.5 * hInPocket.y;
        float pocketDist = length(q.xz);
        dTable = smoothMax(dTable, 0.53 - pocketDist, 0.01);

        // ----------- Cue calculation -----------
        float3 pc = p - float3(balls[0].position.x, 0.05, balls[0].position.y);
        pc.y -= 0.05;

        // The original "baseAngle" for subtle motion:
        float baseAngle = sin(time * 0.5) * 0.1;

        // ---- PITCH and YAW rotations ----
        // 1) Pitch rotation around X axis
        pc = rotateX(pc, cue3DRotate.y);

        // 2) Yaw rotation around the Y axis
        float finalYaw = baseAngle + cue3DRotate.x;  // combine base wobble + user yaw

        // -- FIX: Remove the negative sign so the cue is oriented correctly --
        pc = rotateY(pc, finalYaw);

        // Now shift the cue forward/back based on user pull
        float cueLength = 2.5;
        float ballRadius = 0.47;
        float tipOffset = cueLength;
        float maxCueOffset = -ballRadius;
        pc.z += (maxCueOffset - cueOffset - tipOffset);
        pc = rotateY(pc, 3.14159);

        // Shift horizontally by the user’s offset on the cue tip
        pc.x -= cueTipOffset.x;
        pc.y -= cueTipOffset.y;

        float dCueStick = prRoundCylDf(pc, 0.1 - (0.015 / 2.5) * (pc.z + tipOffset), 0.05, cueLength);
        if (cueVisible == 0) dCueStick = 99999.0;

        float d = min(dTable, dCueStick);
        if (d < 0.0005 || t > dstBall) {
            if (dTable < dCueStick) {
                dstTable = t;
            } else {
                dstCue = t;
                cueHitPos = pc;
            }
            break;
        }
        t += d * 0.7;
        if (t > maxDist) break;
    }

    float3 lightPos = float3(0.0, 3.0 * hbLen, 0.0);
    float minDist = min(min(dstBall, dstTable), dstCue);

    // =============== Shading ================
    if (minDist < maxDist) {
        if (dstBall <= min(dstTable, dstCue)) {
            float3 p = ro + rd * dstBall;
            float3 n = ballNormal;
            int id = ballId;
            if (id == 0) {
                col = float3(1.0);
            } else {
                float c = float(id - 1);
                float3 baseColor;
                bool isStriped = (id >= 9);
                if (id == 8) {
                    baseColor = float3(0.0);
                } else {
                    baseColor = hsvToRgb(float3(fmod(c / 7.0, 1.0), 1.0, 1.0));
                }
                float3x3 rotMat = qtToRMat(balls[id].quaternion);
                float3 rotatedNormal = rotMat * n;
                float2 uv = float2(
                    atan2(rotatedNormal.x, rotatedNormal.z) / (2.0 * PI) + 0.5,
                    acos(rotatedNormal.y) / PI
                );
                if (isStriped && id != 8) {
                    float stripeFactor = sin(uv.y * PI * 10.0);
                    col = mix(float3(1.0), baseColor, step(0.0, stripeFactor));
                } else {
                    col = baseColor;
                }
                float2 circleCenter = float2(0.5, 0.5);
                float circleRadius = 0.2;
                float distToCenter = length(uv - circleCenter);
                if (distToCenter < circleRadius && id != 0) {
                    col = float3(1.0);
                    if (distToCenter < circleRadius * 0.7) col = float3(0.0);
                }
            }
            col *= 0.2 + 0.8 * max(n.y, 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col += float3(0.2) * spec;
        }
        else if (dstCue < dstTable) {
            float3 p = ro + rd * dstCue;
            float3 eps = float3(0.001, 0.0, 0.0);
            float3 n = normalize(float3(
                prRoundCylDf(cueHitPos + eps.xyy, 0.1, 0.05, 2.5) -
                prRoundCylDf(cueHitPos - eps.xyy, 0.1, 0.05, 2.5),
                prRoundCylDf(cueHitPos + eps.yxy, 0.1, 0.05, 2.5) -
                prRoundCylDf(cueHitPos - eps.yxy, 0.1, 0.05, 2.5),
                prRoundCylDf(cueHitPos + eps.yyx, 0.1, 0.05, 2.5) -
                prRoundCylDf(cueHitPos - eps.yyx, 0.1, 0.05, 2.5)
            ));
            // Simple 2-tone color for the cue
            col = (cueHitPos.z < 2.2) ? float3(0.5, 0.3, 0.0) : float3(0.7, 0.7, 0.3);
            float diff = max(dot(n, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col *= 0.3 + 0.7 * diff;
            col += float3(0.2) * spec;
        }
        else {
            float3 p = ro + rd * dstTable;
            float3 eps = float3(0.001, 0.0, 0.0);
            float3 n = normalize(float3(
                prBoxDf(p + eps.xyy, float3(hIn.x, 0.4, hIn.y)) -
                prBoxDf(p - eps.xyy, float3(hIn.x, 0.4, hIn.y)),
                prBoxDf(p + eps.yxy, float3(hIn.x, 0.4, hIn.y)) -
                prBoxDf(p - eps.yxy, float3(hIn.x, 0.4, hIn.y)),
                prBoxDf(p + eps.yyx, float3(hIn.x, 0.4, hIn.y)) -
                prBoxDf(p - eps.yyx, float3(hIn.x, 0.4, hIn.y))
            ));
            float2 pocketCheck = float2(
                abs(p.x) - (hIn.x - bWid + 0.03),
                fmod(p.z + 0.5 * (hIn.y - bWid + 0.03), (hIn.y - bWid + 0.03)) - 0.5 * (hIn.y - bWid + 0.03)
            );
            float pocketDist = length(pocketCheck);
            if (pocketDist < 0.53) {
                col = float3(0.0);
            } else if (max(abs(p.x) - hIn.x, abs(p.z) - hIn.y) < 0.3) {
                col = float3(0.1, 0.5, 0.3);
            } else {
                col = float3(0.3, 0.1, 0.0);
            }
            float diff = max(dot(n, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col *= 0.3 + 0.7 * diff;
            col += float3(0.2) * spec;
        }
    }

    return clamp(col, 0.0, 1.0);
}

// =============== Standard Vertex/Fragment Shaders ================
vertex VertexOut vertexShader(uint vertexID [[vertex_id]]) {
    constexpr float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    constexpr float2 uvs[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0)
    };
    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.uv = uvs[vertexID];
    return out;
}

// Orbiting camera fragment
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant float2 &resolution   [[buffer(0)]],
                               constant float &time          [[buffer(1)]],
                               constant Ball* balls          [[buffer(2)]],
                               constant float &cueOffset     [[buffer(3)]],
                               constant int &cueVisible      [[buffer(4)]],
                               constant float2 &cueTipOffset [[buffer(5)]],
                               constant float &cueAngle      [[buffer(6)]],
                               constant float2 &cue3DRotate  [[buffer(7)]]) {
    float2 uv = 2.0 * in.uv - 1.0;
    uv.x *= resolution.x / resolution.y;

    float3 target = float3(0.0, 0.0, 0.0);
    float3 ro = float3(0.0, 10.0, 20.0);

    float angle = time * 0.1;
    ro = float3(sin(angle) * 20.0, 10.0, cos(angle) * 20.0);

    float3 vd = normalize(target - ro);
    float3 right = normalize(cross(float3(0.0, 1.0, 0.0), vd));
    float3 up = normalize(cross(vd, right));

    const float fov = 0.8;
    float3 rd = normalize(vd + right * uv.x * fov + up * uv.y * fov);

    float3 col = showScene(ro, rd, time, cueOffset, cueTipOffset,
                           balls, cueVisible, cueAngle, cue3DRotate);
    return float4(col, 1.0);
}

// Behind-ball camera fragment
fragment float4 behindBallFragmentShader(VertexOut in [[stage_in]],
                                         constant float2 &resolution   [[buffer(0)]],
                                         constant float3 &cameraPos    [[buffer(1)]],
                                         constant float3 &cameraTarget [[buffer(2)]],
                                         constant Ball*  balls         [[buffer(3)]],
                                         constant float  &cueOffset    [[buffer(4)]],
                                         constant int    &cueVisible   [[buffer(5)]],
                                         constant float2 &cueTipOffset [[buffer(6)]],
                                         constant float  &cueAngle     [[buffer(7)]],
                                         constant float2 &cue3DRotate  [[buffer(8)]]) {
    float2 uv = 2.0 * in.uv - 1.0;
    uv.x *= resolution.x / resolution.y;

    float3 ro = cameraPos;
    float3 target = cameraTarget;

    float3 vd = normalize(target - ro);
    float3 right = normalize(cross(float3(0.0, 1.0, 0.0), vd));
    float3 up = normalize(cross(vd, right));

    const float fov = 0.8;
    float3 rd = normalize(vd + right * uv.x * fov + up * uv.y * fov);

    float timeDummy = 0.0;
    float3 col = showScene(ro, rd, timeDummy, cueOffset, cueTipOffset,
                           balls, cueVisible, cueAngle, cue3DRotate);
    return float4(col, 1.0);
}

// Third-ball camera fragment
fragment float4 thirdBallFragmentShader(VertexOut in [[stage_in]],
                                        constant float2 &resolution   [[buffer(0)]],
                                        constant float3 &cameraPos    [[buffer(1)]],
                                        constant float3 &cameraTarget [[buffer(2)]],
                                        constant Ball*  balls         [[buffer(3)]],
                                        constant float  &cueOffset    [[buffer(4)]],
                                        constant int    &cueVisible   [[buffer(5)]],
                                        constant float2 &cueTipOffset [[buffer(6)]],
                                        constant float  &cueAngle     [[buffer(7)]],
                                        constant float2 &cue3DRotate  [[buffer(8)]]) {
    float2 uv = 2.0 * in.uv - 1.0;
    uv.x *= resolution.x / resolution.y;

    float3 ro = cameraPos;
    float3 target = cameraTarget;

    float3 vd = normalize(target - ro);
    float3 right = normalize(cross(float3(0.0, 1.0, 0.0), vd));
    float3 up = normalize(cross(vd, right));

    const float fov = 0.8;
    float3 rd = normalize(vd + right * uv.x * fov + up * uv.y * fov);

    float timeDummy = 0.0;
    float3 col = showScene(ro, rd, timeDummy, cueOffset, cueTipOffset,
                           balls, cueVisible, cueAngle, cue3DRotate);
    return float4(col, 1.0);
}
"""

// MARK: - Swift Utility for Quaternions
func quaternionFromAxisAngle(_ axis: SIMD3<Float>, _ angle: Float) -> SIMD4<Float> {
    let halfAngle = angle * 0.5
    let s = sin(halfAngle)
    return SIMD4<Float>(axis.x * s, axis.y * s, axis.z * s, cos(halfAngle))
}

func quaternionMultiply(_ q1: SIMD4<Float>, _ q2: SIMD4<Float>) -> SIMD4<Float> {
    SIMD4<Float>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    )
}

// MARK: - Ball Data
struct BallData {
    var position: SIMD2<Float>
    var velocity: SIMD2<Float>
    var quaternion: SIMD4<Float>
}

// MARK: - Main Simulation
final class BilliardSimulation: ObservableObject {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private let orbitPipeline: MTLRenderPipelineState
    private let behindPipeline: MTLRenderPipelineState
    private let thirdPipeline: MTLRenderPipelineState

    @Published var time: Float = 0.0
    @Published var isTouching: Bool = false
    
    // How far the user has pulled the stick backward/forward
    @Published var cueOffset: Float = 0.0
    
    // Tip offset on the cue’s face (where you strike the ball)
    @Published var cueTipOffset: SIMD2<Float> = .zero
    
    // Whether we should see the cue in the scene
    @Published var showCueValue: Int32 = 1
    
    // Original single angle (used by “behind” camera or orbit)
    @Published var cueAngle: Float = 0.0
    
    // New Yaw/Pitch for full 3D orientation
    @Published var cue3DRotate: SIMD2<Float> = SIMD2<Float>(0, 0)
    
    @Published var shooting: Bool = false

    var balls: [BallData]
    private var ballBuffer: MTLBuffer

    // Physics config
    private let ballRadius: Float = 0.47
    private let tableWidth: Float = 7.6
    private let tableLength: Float = 13.3
    private let pocketRadius: Float = 0.53
    private let cuePullSpeed: Float = 1.0
    private let cueStrikeSpeed: Float = 5.0
    private let maxCueOffset: Float = 2.0
    public  let maxTipOffset: Float = 0.47

    private var hitTriggered: Bool = false
    private var powerAtRelease: Float = 0.0

    private let friction: Float = 0.96
    private let restitution: Float = 0.75
    private let minVelocity: Float = 0.005
    private let ballMass: Float = 1.0

    // Uniform buffers
    private var resolution = SIMD2<Float>(0,0)
    private var orbitUniformsBuffer: MTLBuffer
    private var behindCamPosBuffer: MTLBuffer
    private var behindCamTargetBuffer: MTLBuffer
    private var thirdCamPosBuffer: MTLBuffer
    private var thirdCamTargetBuffer: MTLBuffer
    private var cueOffsetBuffer: MTLBuffer
    private var showCueBuffer: MTLBuffer
    private var cueTipOffsetBuffer: MTLBuffer
    private var cueAngleBuffer: MTLBuffer
    private var cue3DRotateBuffer: MTLBuffer

    init?() {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let cq = dev.makeCommandQueue() else {
            return nil
        }
        device = dev
        commandQueue = cq

        // Compile our Metal shaders
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: metalShader, options: nil)
        } catch {
            print("Failed to create Metal library: \(error)")
            return nil
        }

        guard
            let vertexFunction    = library.makeFunction(name: "vertexShader"),
            let orbitFragment     = library.makeFunction(name: "fragmentShader"),
            let behindFragment    = library.makeFunction(name: "behindBallFragmentShader"),
            let thirdFragment     = library.makeFunction(name: "thirdBallFragmentShader")
        else {
            print("Missing required shader functions.")
            return nil
        }

        do {
            // Orbit pipeline
            let orbitDescriptor = MTLRenderPipelineDescriptor()
            orbitDescriptor.vertexFunction = vertexFunction
            orbitDescriptor.fragmentFunction = orbitFragment
            orbitDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            orbitPipeline = try device.makeRenderPipelineState(descriptor: orbitDescriptor)

            // Behind pipeline
            let behindDescriptor = MTLRenderPipelineDescriptor()
            behindDescriptor.vertexFunction = vertexFunction
            behindDescriptor.fragmentFunction = behindFragment
            behindDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            behindPipeline = try device.makeRenderPipelineState(descriptor: behindDescriptor)

            // Third pipeline
            let thirdDescriptor = MTLRenderPipelineDescriptor()
            thirdDescriptor.vertexFunction = vertexFunction
            thirdDescriptor.fragmentFunction = thirdFragment
            thirdDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            thirdPipeline = try device.makeRenderPipelineState(descriptor: thirdDescriptor)

        } catch {
            print("Failed to create pipeline state: \(error)")
            return nil
        }

        let identityQuat = SIMD4<Float>(0, 0, 0, 1)
        self.balls = [
            // White ball in front
            BallData(position: SIMD2<Float>( 0.0,  5.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            // Triangular rack
            BallData(position: SIMD2<Float>(-0.5, -2.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 0.5, -2.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-1.0, -1.5), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 0.0, -1.5), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 1.0, -1.5), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-1.5, -1.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-0.5, -1.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 0.5, -1.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 1.5, -1.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-2.0, -0.5), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-1.0, -0.5), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 0.0, -0.5), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 1.0, -0.5), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 2.0, -0.5), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
            BallData(position: SIMD2<Float>( 0.0,  0.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat)
        ]

        // Prepare ball buffer for GPU
        var ballShaderData = [SIMD8<Float>](repeating: SIMD8<Float>(0,0,0,0,0,0,0,0), count: 16)
        for i in 0..<16 {
            ballShaderData[i] = SIMD8<Float>(
                balls[i].position.x,  balls[i].position.y,
                balls[i].velocity.x,  balls[i].velocity.y,
                balls[i].quaternion.x, balls[i].quaternion.y,
                balls[i].quaternion.z, balls[i].quaternion.w
            )
        }
        self.ballBuffer = device.makeBuffer(bytes: ballShaderData,
                                            length: MemoryLayout<SIMD8<Float>>.stride * 16,
                                            options: .storageModeShared)!

        orbitUniformsBuffer     = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        behindCamPosBuffer      = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        behindCamTargetBuffer   = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        thirdCamPosBuffer       = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        thirdCamTargetBuffer    = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!

        cueOffsetBuffer         = device.makeBuffer(length: MemoryLayout<Float>.stride,  options: .storageModeShared)!
        showCueBuffer           = device.makeBuffer(length: MemoryLayout<Int32>.stride,  options: .storageModeShared)!
        cueTipOffsetBuffer      = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        cueAngleBuffer          = device.makeBuffer(length: MemoryLayout<Float>.stride,  options: .storageModeShared)!
        cue3DRotateBuffer       = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
    }

    private func checkPocket(pos: SIMD2<Float>) -> Bool {
        let pocketPositions: [SIMD2<Float>] = [
            SIMD2<Float>(-tableWidth + ballRadius, -tableLength + ballRadius),
            SIMD2<Float>( tableWidth - ballRadius, -tableLength + ballRadius),
            SIMD2<Float>(-tableWidth + ballRadius,  0.0),
            SIMD2<Float>( tableWidth - ballRadius,  0.0),
            SIMD2<Float>(-tableWidth + ballRadius,  tableLength - ballRadius),
            SIMD2<Float>( tableWidth - ballRadius,  tableLength - ballRadius),
        ]
        for p in pocketPositions {
            if simd_length(pos - p) < pocketRadius {
                return true
            }
        }
        return false
    }

    // =============== PHYSICS UPDATE ===============
    func updatePhysics(deltaTime: Float) {
        // Cue pulling and releasing
        if isTouching && !shooting {
            cueOffset += cuePullSpeed * deltaTime
            if cueOffset > maxCueOffset {
                cueOffset = maxCueOffset
            }
        }
        else if !isTouching && cueOffset > 0.0 {
            if !shooting {
                powerAtRelease = cueOffset / maxCueOffset
                shooting = true
            }
            cueOffset -= cueStrikeSpeed * deltaTime
            if (cueOffset <= 0.0) {
                cueOffset = 0.0
                let baseVelocity: Float = -20.0
                let velocityScale = 0.3 + 1.7 * powerAtRelease
                // For simplicity, applying the velocity using cueAngle only:
                let angleRad = -cueAngle
                let cosA = cos(angleRad)
                let sinA = sin(angleRad)

                var velocity = SIMD2<Float>(0, baseVelocity * velocityScale)
                velocity = SIMD2<Float>(
                    velocity.x * cosA - velocity.y * sinA,
                    velocity.x * sinA + velocity.y * cosA
                )
                balls[0].velocity = velocity

                hitTriggered = true
                shooting = false
                powerAtRelease = 0.0
                showCueValue = 0
            }
        }

        if !isTouching && hitTriggered {
            var allStopped = true
            for ball in balls {
                if ball.velocity.x.isInfinite { continue }
                if simd_length(ball.velocity) > minVelocity {
                    allStopped = false
                    break
                }
            }
            if allStopped {
                hitTriggered = false
                showCueValue = 1
            }
        }

        // Basic sub-stepping
        let subSteps = 4
        let dt = deltaTime / Float(subSteps)

        for _ in 0..<subSteps {
            for i in 0..<16 {
                var pos = balls[i].position
                var vel = balls[i].velocity

                if vel.x.isInfinite { continue }

                pos += vel * dt

                if checkPocket(pos: pos) {
                    // Pocketed
                    vel = SIMD2<Float>(.infinity, .infinity)
                    pos = SIMD2<Float>(0.0, 0.0)
                } else {
                    if abs(pos.x) > tableWidth - ballRadius {
                        pos.x = (pos.x > 0) ? tableWidth - ballRadius : -tableWidth + ballRadius
                        vel.x = -vel.x * restitution
                    }
                    if abs(pos.y) > tableLength - ballRadius {
                        pos.y = (pos.y > 0) ? tableLength - ballRadius : -tableLength + ballRadius
                        vel.y = -vel.y * restitution
                    }
                }

                vel *= friction
                if simd_length(vel) < minVelocity {
                    vel = .zero
                }

                balls[i].position = pos
                balls[i].velocity = vel
            }

            // Ball-ball collisions
            for i in 0..<15 {
                for j in (i+1)..<16 {
                    var pos1 = balls[i].position
                    var pos2 = balls[j].position
                    var vel1 = balls[i].velocity
                    var vel2 = balls[j].velocity

                    if vel1.x.isInfinite || vel2.x.isInfinite { continue }

                    let delta = pos2 - pos1
                    let dist = simd_length(delta)
                    if dist < (2.0 * ballRadius) {
                        let normal = delta / dist
                        let relativeVel = vel1 - vel2
                        let impulse = simd_dot(relativeVel, normal)
                        if impulse > 0.0 {
                            let impulseMag = -(1.0 + restitution) * impulse / (2.0 / ballMass)
                            vel1 += normal * (impulseMag / ballMass)
                            vel2 -= normal * (impulseMag / ballMass)

                            let overlap = (2.0 * ballRadius) - dist
                            let correction = normal * (overlap * 0.5)
                            pos1 -= correction
                            pos2 += correction
                        }
                        balls[i].position = pos1
                        balls[j].position = pos2
                        balls[i].velocity = vel1
                        balls[j].velocity = vel2
                    }
                }
            }
        }

        // Write back to GPU buffer
        let ptr = ballBuffer.contents().bindMemory(to: SIMD8<Float>.self, capacity: 16)
        for i in 0..<16 {
            ptr[i] = SIMD8<Float>(
                balls[i].position.x,  balls[i].position.y,
                balls[i].velocity.x,  balls[i].velocity.y,
                balls[i].quaternion.x, balls[i].quaternion.y,
                balls[i].quaternion.z, balls[i].quaternion.w
            )
        }
    }

    // MARK: - Encode Renders

    func encodeOrbitRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        // Buffers to match the fragmentShader signature
        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)

        let timePtr = orbitUniformsBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        timePtr[0] = self.time
        encoder.setFragmentBuffer(orbitUniformsBuffer, offset: 0, index: 1)

        encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 2)

        cueOffsetBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueOffset
        encoder.setFragmentBuffer(cueOffsetBuffer, offset: 0, index: 3)

        showCueBuffer.contents().bindMemory(to: Int32.self, capacity: 1)[0] = showCueValue
        encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 4)

        cueTipOffsetBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cueTipOffset
        encoder.setFragmentBuffer(cueTipOffsetBuffer, offset: 0, index: 5)

        cueAngleBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueAngle
        encoder.setFragmentBuffer(cueAngleBuffer, offset: 0, index: 6)

        cue3DRotateBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cue3DRotate
        encoder.setFragmentBuffer(cue3DRotateBuffer, offset: 0, index: 7)

        encoder.setRenderPipelineState(orbitPipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    func encodeBehindRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0, 2.0, 0)
        var cameraTarget = SIMD3<Float>(0, 0.0, 0)
        let speed = simd_length(whiteBall.velocity)

        // Simple logic: if not moving, aim behind the ball using cueAngle
        if speed < 0.01 {
            cameraTarget = SIMD3<Float>(whiteBall.position.x, 0.05, whiteBall.position.y)
            let stationaryDistance: Float = 2.5
            let angleRad = -cueAngle
            let cosA = cos(angleRad)
            let sinA = sin(angleRad)
            let defaultOffset = SIMD3<Float>(0, 0.7, stationaryDistance)
            let rotatedOffset = SIMD3<Float>(
                defaultOffset.x * cosA - defaultOffset.z * sinA,
                defaultOffset.y,
                defaultOffset.x * sinA + defaultOffset.z * cosA
            )
            cameraPosition = cameraTarget + rotatedOffset
        } else {
            let forward = simd_normalize(SIMD3<Float>(whiteBall.velocity.x, 0, whiteBall.velocity.y) + 0.000001)
            cameraTarget = SIMD3<Float>(whiteBall.position.x, 0.05, whiteBall.position.y)
            cameraPosition = cameraTarget - (forward * 3.0)
            cameraPosition.y += 1.0
        }

        // Set buffers
        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        behindCamPosBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)[0] = cameraPosition
        encoder.setFragmentBuffer(behindCamPosBuffer, offset: 0, index: 1)

        behindCamTargetBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)[0] = cameraTarget
        encoder.setFragmentBuffer(behindCamTargetBuffer, offset: 0, index: 2)

        encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 3)

        cueOffsetBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueOffset
        encoder.setFragmentBuffer(cueOffsetBuffer, offset: 0, index: 4)

        showCueBuffer.contents().bindMemory(to: Int32.self, capacity: 1)[0] = showCueValue
        encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 5)

        cueTipOffsetBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cueTipOffset
        encoder.setFragmentBuffer(cueTipOffsetBuffer, offset: 0, index: 6)

        cueAngleBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueAngle
        encoder.setFragmentBuffer(cueAngleBuffer, offset: 0, index: 7)

        cue3DRotateBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cue3DRotate
        encoder.setFragmentBuffer(cue3DRotateBuffer, offset: 0, index: 8)

        encoder.setRenderPipelineState(behindPipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    func encodeThirdRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0, 2.0, 0)
        var cameraTarget = SIMD3<Float>(0, 0.0, 0)
        let speed = simd_length(whiteBall.velocity)

        // If ball is still, place camera behind the ball, offset more
        if speed < 0.01 {
            cameraTarget = SIMD3<Float>(whiteBall.position.x, 0.05, whiteBall.position.y)
            let stationaryDistance: Float = 7.0
            let angleRad = -cueAngle
            let cosA = cos(angleRad)
            let sinA = sin(angleRad)
            let defaultOffset = SIMD3<Float>(0, 0.7, stationaryDistance)
            let rotatedOffset = SIMD3<Float>(
                defaultOffset.x * cosA - defaultOffset.z * sinA,
                defaultOffset.y,
                defaultOffset.x * sinA + defaultOffset.z * cosA
            )
            cameraPosition = cameraTarget + rotatedOffset
        } else {
            let forward = simd_normalize(SIMD3<Float>(whiteBall.velocity.x, 0, whiteBall.velocity.y) + 0.000001)
            cameraTarget = SIMD3<Float>(whiteBall.position.x, 0.05, whiteBall.position.y)
            let movingDistance: Float = 8.0
            cameraPosition = cameraTarget - (forward * movingDistance)
            cameraPosition.y += 1.0
        }

        // Set buffers
        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        thirdCamPosBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)[0] = cameraPosition
        encoder.setFragmentBuffer(thirdCamPosBuffer, offset: 0, index: 1)

        thirdCamTargetBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)[0] = cameraTarget
        encoder.setFragmentBuffer(thirdCamTargetBuffer, offset: 0, index: 2)

        encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 3)

        cueOffsetBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueOffset
        encoder.setFragmentBuffer(cueOffsetBuffer, offset: 0, index: 4)

        showCueBuffer.contents().bindMemory(to: Int32.self, capacity: 1)[0] = showCueValue
        encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 5)

        cueTipOffsetBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cueTipOffset
        encoder.setFragmentBuffer(cueTipOffsetBuffer, offset: 0, index: 6)

        cueAngleBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueAngle
        encoder.setFragmentBuffer(cueAngleBuffer, offset: 0, index: 7)

        cue3DRotateBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cue3DRotate
        encoder.setFragmentBuffer(cue3DRotateBuffer, offset: 0, index: 8)

        encoder.setRenderPipelineState(thirdPipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
}

// MARK: - SwiftUI MTKViews
struct OrbitingMetalView: UIViewRepresentable {
    @ObservedObject var simulation: BilliardSimulation

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: OrbitingMetalView
        init(_ parent: OrbitingMetalView) { self.parent = parent }
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let commandBuffer = parent.simulation.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
            else { return }

            parent.simulation.encodeOrbitRenderPass(encoder: encoder, viewSize: view.drawableSize)
            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: simulation.device)
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}

struct BehindBallMetalView: UIViewRepresentable {
    @ObservedObject var simulation: BilliardSimulation

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: BehindBallMetalView
        init(_ parent: BehindBallMetalView) { self.parent = parent }
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let commandBuffer = parent.simulation.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
            else { return }

            parent.simulation.encodeBehindRenderPass(encoder: encoder, viewSize: view.drawableSize)
            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: simulation.device)
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}

struct ThirdBallMetalView: UIViewRepresentable {
    @ObservedObject var simulation: BilliardSimulation

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: ThirdBallMetalView
        init(_ parent: ThirdBallMetalView) { self.parent = parent }
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let commandBuffer = parent.simulation.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
            else { return }

            parent.simulation.encodeThirdRenderPass(encoder: encoder, viewSize: view.drawableSize)
            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: simulation.device)
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}

// MARK: - ContentView
struct ContentView: View {
    @StateObject private var simulation = BilliardSimulation()!

    // For behind-ball (manipulating tip offset)
    @State private var viewSizeBehind: CGSize = .zero
    @State private var initialTouchBehind: CGPoint? = nil
    @State private var initialTipOffsetBehind: SIMD2<Float> = .zero

    // For third-ball (manipulating 3D angles)
    @State private var viewSizeThird: CGSize = .zero
    @State private var initialTouchThird: CGPoint? = nil
    @State private var initialCueYawThird: Float = 0.0
    @State private var initialCuePitchThird: Float = 0.0

    var body: some View {
        ZStack {
            // Full-screen orbit
            OrbitingMetalView(simulation: simulation)
                .edgesIgnoringSafeArea(.all)
                .overlay(
                    Text("Orbiting Camera")
                        .foregroundColor(.white)
                        .padding(),
                    alignment: .top
                )
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in
                            simulation.isTouching = true
                        }
                        .onEnded { _ in
                            simulation.isTouching = false
                        }
                )

            VStack {
                Spacer()
                HStack {
                    // Behind-Ball camera
                    BehindBallMetalView(simulation: simulation)
                        .frame(width: UIScreen.main.bounds.width / 2,
                               height: UIScreen.main.bounds.width / 2)
                        .background(
                            GeometryReader { geo in
                                Color.clear
                                    .onAppear { viewSizeBehind = geo.size }
                                    .onChange(of: geo.size) { newSize in
                                        viewSizeBehind = newSize
                                    }
                            }
                        )
                        .overlay(
                            Text("Behind-Ball Camera")
                                .foregroundColor(.white)
                                .padding(),
                            alignment: .top
                        )
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    if simulation.showCueValue == 1 && !simulation.shooting {
                                        if initialTouchBehind == nil {
                                            initialTouchBehind = value.startLocation
                                            initialTipOffsetBehind = simulation.cueTipOffset
                                        }
                                        guard let start = initialTouchBehind else { return }

                                        let deltaX = Float(value.location.x - start.x)
                                        let deltaY = Float(value.location.y - start.y)

                                        let scaleFactor = simulation.maxTipOffset / Float(viewSizeBehind.height) * 2.0
                                        let aspect = Float(viewSizeBehind.width / viewSizeBehind.height)

                                        var newOffset = initialTipOffsetBehind + SIMD2<Float>(
                                            deltaX * scaleFactor * aspect,
                                            -deltaY * scaleFactor + 0.05
                                        )
                                        // clamp radius
                                        let offsetLength = simd_length(newOffset)
                                        if offsetLength > simulation.maxTipOffset {
                                            newOffset *= simulation.maxTipOffset / offsetLength
                                        }
                                        simulation.cueTipOffset = newOffset
                                    }
                                }
                                .onEnded { _ in
                                    initialTouchBehind = nil
                                    initialTipOffsetBehind = .zero
                                }
                        )

                    // Third-Ball camera (3D rotation)
                    ThirdBallMetalView(simulation: simulation)
                        .frame(width: UIScreen.main.bounds.width / 2,
                               height: UIScreen.main.bounds.width / 2)
                        .background(
                            GeometryReader { geo in
                                Color.clear
                                    .onAppear { viewSizeThird = geo.size }
                                    .onChange(of: geo.size) { newSize in
                                        viewSizeThird = newSize
                                    }
                            }
                        )
                        .overlay(
                            Text("Third-Ball Camera (3D Cue Rotation)")
                                .foregroundColor(.white)
                                .padding(),
                            alignment: .top
                        )
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    if simulation.showCueValue == 1 && !simulation.shooting {
                                        if initialTouchThird == nil {
                                            initialTouchThird = value.startLocation
                                            initialCueYawThird = simulation.cue3DRotate.x
                                            initialCuePitchThird = simulation.cue3DRotate.y
                                        }
                                        guard let start = initialTouchThird else { return }

                                        let deltaX = Float(value.location.x - start.x)
                                        let deltaY = Float(value.location.y - start.y)
                                        let sensitivity: Float = 0.01

                                        // Yaw changes with X movement
                                        let newYaw = initialCueYawThird + deltaX * sensitivity
                                        // Pitch changes with Y movement
                                        let newPitch = initialCuePitchThird - deltaY * sensitivity

                                        // Example clamp for pitch so it doesn't flip over
                                        let clampedPitch = max(-0.8, min(0.8, newPitch))

                                        simulation.cue3DRotate = SIMD2<Float>(newYaw, clampedPitch)
                                    }
                                }
                                .onEnded { _ in
                                    initialTouchThird = nil
                                }
                        )
                }
                .padding(.bottom, 20)
            }
        }
        .onAppear {
            // Simple game loop
            let timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { _ in
                simulation.time += 1.0/60.0
                simulation.updatePhysics(deltaTime: 1.0/60.0)
            }
            RunLoop.current.add(timer, forMode: .common)
        }
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
