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

// For intersecting the balls:
void ballHit(float3 ro, float3 rd, thread float &dist, thread float3 &normal,
             thread int &id, constant Ball* balls [[buffer(2)]]) {
    const int nBall = 16;
    const float rad = 0.47; // Ball radius
    dist = 50.0;
    normal = float3(0.0);
    id = -1;
    for (int n = 0; n < nBall; n++) {
        // All balls have center at y=0.47
        float3 u = ro - float3(balls[n].position.x, 0.47, balls[n].position.y);
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

// Renders geometry (table, pockets, balls, cue).
float3 showScene(float3 ro, float3 rd,
                 float time,
                 float cueOffset,
                 float2 cueTipOffset,
                 constant Ball* balls [[buffer(2)]],
                 int cueVisible,
                 float cueAngle,
                 float2 cue3DRotate)
{
    const float hbLen = 8.0;
    const float bWid  = 0.4;
    const float2 hIn  = float2(hbLen, hbLen * 1.75) - bWid;
    const float PI    = 3.14159;

    float3 col = float3(0.05, 0.05, 0.1);
    float t = 0.0;
    const float maxDist = 50.0;

    // Ball intersection
    float dstBall;
    float3 ballNormal;
    int ballId;
    ballHit(ro, rd, dstBall, ballNormal, ballId, balls);

    // Table/cue intersection distances
    float dstTable = maxDist;
    float dstCue = maxDist;
    float3 cueHitPos;

    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * t;

        // Table geometry
        float dSurface = prBoxDf(p, float3(hIn.x, 0.4, hIn.y));
        float3 pb = p;
        pb.y -= -0.6;  // The table top is at y=0, but we use a box offset
        float dBorder = prRoundBoxDf(pb, float3(hIn.x + 0.6, 0.5, hIn.y + 0.6), 0.2);
        float dTable = max(dBorder, -dSurface);

        // Pockets
        float2 hInPocket = hIn - bWid + 0.03;
        float3 q = p;
        q.x = abs(q.x) - hInPocket.x;
        q.z = fmod(q.z + 0.5 * hInPocket.y, hInPocket.y) - 0.5 * hInPocket.y;
        float pocketDist = length(q.xz);
        dTable = smoothMax(dTable, 0.53 - pocketDist, 0.01);

        // Cue geometry (white ball is index 0)
        float3 pc = p - float3(balls[0].position.x, 0.47, balls[0].position.y);
        float baseAngle = sin(time * 0.5) * 0.1;
        pc = rotateX(pc, cue3DRotate.y);
        float finalYaw = baseAngle + cue3DRotate.x;
        pc = rotateY(pc, finalYaw);

        float cueLength = 2.5;
        float ballRadius = 0.47;
        float tipOffset = cueLength;
        float maxCueOffset = -ballRadius;
        pc.z += (maxCueOffset - cueOffset - tipOffset);
        // Flip it around so the cue extends behind the ball
        pc = rotateY(pc, 3.14159);

        // X/Y offset from user drag
        pc.x -= cueTipOffset.x;
        pc.y -= cueTipOffset.y;

        // Render the cue as a "rounded cylinder"
        float dCueStick = prRoundCylDf(pc, 0.1 - (0.015 / 2.5) * (pc.z + tipOffset), 0.05, cueLength);
        if (cueVisible == 0) {
            dCueStick = 99999.0;
        }

        // Combine table & cue distances
        float d = min(dTable, dCueStick);

        // Break if we hit table/cue or if we've gone farther than the nearest ball
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
    if (minDist < maxDist) {
        // Ball shading
        if (dstBall <= min(dstTable, dstCue)) {
            float3 p = ro + rd * dstBall;
            float3 n = ballNormal;
            int id = ballId;
            if (id == 0) {
                // White ball
                col = float3(1.0);
            } else {
                // Colored balls
                float c = float(id - 1);
                float3 baseColor;
                bool isStriped = (id >= 9);
                if (id == 8) {
                    // 8-ball is black
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
                // Stripe pattern for balls 9..15
                if (isStriped && id != 8) {
                    float stripeFactor = sin(uv.y * PI * 10.0);
                    col = mix(float3(1.0), baseColor, step(0.0, stripeFactor));
                } else {
                    col = baseColor;
                }
                // Number circle
                float2 circleCenter = float2(0.5, 0.5);
                float circleRadius = 0.2;
                float distToCenter = length(uv - circleCenter);
                if (distToCenter < circleRadius && id != 0) {
                    // White circle
                    col = float3(1.0);
                    // Number area in black
                    if (distToCenter < circleRadius * 0.7) col = float3(0.0);
                }
            }
            // Light & specular
            col *= 0.2 + 0.8 * max(n.y, 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col += float3(0.2) * spec;
        }
        // Cue shading
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
            // Two-tone cue color
            col = (cueHitPos.z < 2.2) ? float3(0.5, 0.3, 0.0) : float3(0.7, 0.7, 0.3);

            float diff = max(dot(n, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col *= 0.3 + 0.7 * diff;
            col += float3(0.2) * spec;
        }
        // Table shading
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
                // Pocket
                col = float3(0.0);
            } else if (max(abs(p.x) - hIn.x, abs(p.z) - hIn.y) < 0.3) {
                // Felt
                col = float3(0.1, 0.5, 0.3);
            } else {
                // Wooden rails
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

// Simple pass-thru vertex
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

    // Orbiting camera
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

// MARK: - Swift Utility Functions
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
    var angularVelocity: SIMD3<Float>
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
    @Published var cueOffset: Float = 0.0
    @Published var cueTipOffset: SIMD2<Float> = .zero
    @Published var showCueValue: Int32 = 1
    @Published var cueAngle: Float = 0.0
    @Published var cue3DRotate: SIMD2<Float> = SIMD2<Float>(0, 0)
    @Published var shooting: Bool = false

    var balls: [BallData]
    private var ballBuffer: MTLBuffer

    // Physics constants
    private let ballRadius: Float = 0.47
    private let tableWidth: Float = 7.6
    private let tableLength: Float = 13.3
    private let pocketRadius: Float = 0.53
    private let cuePullSpeed: Float = 1.0
    private let cueStrikeSpeed: Float = 5.0
    private let maxCueOffset: Float = 2.0
    public let maxTipOffset: Float = 0.47
    private let ballMass: Float = 0.17
    private let momentOfInertia: Float = 0.4 * 0.17 * 0.47 * 0.47
    private let gravity: Float = 9.81
    private let frictionKinetic: Float = 0.2
    private let frictionRolling: Float = 0.01
    private let frictionSpinDecay: Float = 5.0
    private let restitutionBall: Float = 0.95
    private let restitutionCushion: Float = 0.8
    private let ballFriction: Float = 0.05

    private var hitTriggered: Bool = false
    private var powerAtRelease: Float = 0.0

    // Buffers
    private var resolution = SIMD2<Float>(0, 0)
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

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: metalShader, options: nil)
        } catch {
            print("Failed to create Metal library: \(error)")
            return nil
        }

        guard let vertexFunction = library.makeFunction(name: "vertexShader"),
              let orbitFragment = library.makeFunction(name: "fragmentShader"),
              let behindFragment = library.makeFunction(name: "behindBallFragmentShader"),
              let thirdFragment = library.makeFunction(name: "thirdBallFragmentShader") else {
            print("Missing required shader functions.")
            return nil
        }

        do {
            let orbitDescriptor = MTLRenderPipelineDescriptor()
            orbitDescriptor.vertexFunction = vertexFunction
            orbitDescriptor.fragmentFunction = orbitFragment
            orbitDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            orbitPipeline = try device.makeRenderPipelineState(descriptor: orbitDescriptor)

            let behindDescriptor = MTLRenderPipelineDescriptor()
            behindDescriptor.vertexFunction = vertexFunction
            behindDescriptor.fragmentFunction = behindFragment
            behindDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            behindPipeline = try device.makeRenderPipelineState(descriptor: behindDescriptor)

            let thirdDescriptor = MTLRenderPipelineDescriptor()
            thirdDescriptor.vertexFunction = vertexFunction
            thirdDescriptor.fragmentFunction = thirdFragment
            thirdDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            thirdPipeline = try device.makeRenderPipelineState(descriptor: thirdDescriptor)
        } catch {
            print("Failed to create pipeline state: \(error)")
            return nil
        }

        // Initialize 16 balls; all will render at y=0.47
        let identityQuat = SIMD4<Float>(0, 0, 0, 1)
        self.balls = [
            BallData(position: SIMD2<Float>(0.0, 5.0), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-0.5, -2.0), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(0.5, -2.0), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-1.0, -1.5), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(0.0, -1.5), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(1.0, -1.5), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-1.5, -1.0), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-0.5, -1.0), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(0.5, -1.0), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(1.5, -1.0), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-2.0, -0.5), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-1.0, -0.5), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(0.0, -0.5), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(1.0, -0.5), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(2.0, -0.5), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(0.0, 0.0), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat)
        ]

        var ballShaderData = [SIMD8<Float>](repeating: SIMD8<Float>(0,0,0,0,0,0,0,0), count: 16)
        for i in 0..<16 {
            ballShaderData[i] = SIMD8<Float>(
                balls[i].position.x, balls[i].position.y,
                balls[i].velocity.x, balls[i].velocity.y,
                balls[i].quaternion.x, balls[i].quaternion.y,
                balls[i].quaternion.z, balls[i].quaternion.w
            )
        }
        self.ballBuffer = device.makeBuffer(bytes: ballShaderData,
                                            length: MemoryLayout<SIMD8<Float>>.stride * 16,
                                            options: .storageModeShared)!

        orbitUniformsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        behindCamPosBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        behindCamTargetBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        thirdCamPosBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        thirdCamTargetBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        cueOffsetBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        showCueBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared)!
        cueTipOffsetBuffer = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        cueAngleBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        cue3DRotateBuffer = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
    }

    // Collision check for cue vs. table/balls
    func isCueColliding(offset: Float,
                        tipOffset: SIMD2<Float>,
                        yaw: Float,
                        pitch: Float) -> Bool
    {
        let segments = 20
        let cueLength: Float = 2.5
        let cueRadius: Float = 0.05

        // Matches hIn from showScene
        let bWid: Float = 0.4
        let hbLen: Float = 8.0
        let hIn = SIMD2<Float>(hbLen, hbLen * 1.75) - bWid

        func prBoxDf(_ p: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
            let d = SIMD3<Float>(abs(p.x), abs(p.y), abs(p.z)) - b
            return min(max(d.x, max(d.y, d.z)), 0.0) + simd_length(max(d, SIMD3<Float>(repeating: 0)))
        }
        func prRoundBoxDf(_ p: SIMD3<Float>, _ b: SIMD3<Float>, _ r: Float) -> Float {
            let q = SIMD3<Float>(abs(p.x), abs(p.y), abs(p.z)) - b
            let qmax = SIMD3<Float>(max(q.x, 0), max(q.y, 0), max(q.z, 0))
            return simd_length(qmax) - r + min(max(q.x, max(q.y, q.z)), 0)
        }

        // White ball is index 0, center at y=0.47
        let whiteBallPos = SIMD3<Float>(balls[0].position.x, 0.47, balls[0].position.y)

        for i in 0...segments {
            let frac = Float(i) / Float(segments)
            let zLocal = frac * cueLength
            var p = SIMD3<Float>(0, 0, -zLocal)

            // Apply pitch
            let cp = cos(pitch), sp = sin(pitch)
            let px = p.x
            let py = p.y * cp - p.z * sp
            let pz = p.y * sp + p.z * cp
            p = SIMD3<Float>(px, py, pz)

            // Apply yaw
            let cy = cos(-yaw), sy = sin(-yaw)
            let tmpx = p.x * cy + p.z * sy
            let tmpz = -p.x * sy + p.z * cy
            p.x = tmpx
            p.z = tmpz

            // Shift by tip offset
            p.x -= tipOffset.x
            p.y -= tipOffset.y

            let tipOffsetBase: Float = cueLength
            let maxCueOffsetWorld: Float = 2.0
            p.z += ( -ballRadius - offset - tipOffsetBase + maxCueOffsetWorld )

            // Place around white ball
            p += whiteBallPos

            // --- Table check ---
            let dSurface = prBoxDf(p, SIMD3<Float>(hIn.x, 0.4, hIn.y))
            var pb = p
            pb.y += 0.6
            let dBorder = prRoundBoxDf(pb, SIMD3<Float>(hIn.x + 0.6, 0.5, hIn.y + 0.6), 0.2)
            let dTable = max(dBorder, -dSurface)
            if dTable < -cueRadius {
                return true
            }

            // --- Ball collision check (all balls, each at y=0.47) ---
            for bIdx in 0..<16 {
                if balls[bIdx].velocity.x.isInfinite { continue } // pocketed
                let center = SIMD3<Float>(balls[bIdx].position.x, 0.47, balls[bIdx].position.y)
                let dist = simd_length(p - center)
                if dist < (ballRadius + cueRadius) {
                    return true
                }
            }
        }
        return false
    }

    private func checkPocket(pos: SIMD2<Float>) -> Bool {
        // Simplified pockets at corners/midpoints
        let pocketPositions: [SIMD2<Float>] = [
            SIMD2<Float>(-tableWidth + ballRadius, -tableLength + ballRadius),
            SIMD2<Float>( tableWidth - ballRadius, -tableLength + ballRadius),
            SIMD2<Float>(-tableWidth + ballRadius, 0.0),
            SIMD2<Float>( tableWidth - ballRadius, 0.0),
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

    // Main physics update
    func updatePhysics(deltaTime: Float) {
        // Store old offset for collision revert
        let oldCueOffset = cueOffset

        // Cue pull/strike logic
        if isTouching && !shooting {
            cueOffset += cuePullSpeed * deltaTime
            if cueOffset > maxCueOffset { cueOffset = maxCueOffset }

            // Revert if collision
            if isCueColliding(offset: cueOffset,
                              tipOffset: cueTipOffset,
                              yaw: cue3DRotate.x,
                              pitch: cue3DRotate.y) {
                cueOffset = oldCueOffset
            }
        } else if !isTouching && cueOffset > 0.0 {
            if !shooting {
                powerAtRelease = cueOffset / maxCueOffset
                shooting = true
            }
            cueOffset -= cueStrikeSpeed * deltaTime
            if cueOffset <= 0.0 {
                cueOffset = 0.0
                applyCueStrike()
                shooting = false
                hitTriggered = true
                showCueValue = 0
            } else {
                if isCueColliding(offset: cueOffset,
                                  tipOffset: cueTipOffset,
                                  yaw: cue3DRotate.x,
                                  pitch: cue3DRotate.y) {
                    cueOffset = oldCueOffset
                }
            }
        }

        // Re-show the cue if everything has stopped
        if !isTouching && hitTriggered {
            var allStopped = true
            for ball in balls {
                if simd_length(ball.velocity) > 0.005 || simd_length(ball.angularVelocity) > 0.1 {
                    allStopped = false
                    break
                }
            }
            if allStopped {
                hitTriggered = false
                showCueValue = 1
            }
        }

        // Subdivide steps
        let subSteps = 8
        let dt = deltaTime / Float(subSteps)

        // Integrate in substeps
        for _ in 0..<subSteps {
            for i in 0..<16 {
                var ball = balls[i]
                // Infinity => pocketed
                if ball.velocity.x.isInfinite { continue }

                let v = ball.velocity
                let w = ball.angularVelocity
                let vMag = simd_length(v)

                // Rolling vs. sliding friction
                let relativeVelocityAtContact = v - ballRadius * SIMD2<Float>(-w.z, w.x)
                let sliding = simd_length(relativeVelocityAtContact) > 0.001

                if sliding {
                    let frictionDir = -simd_normalize(relativeVelocityAtContact)
                    let frictionForce = frictionKinetic * ballMass * gravity
                    ball.velocity += (frictionForce / ballMass) * frictionDir * dt

                    let torque = frictionForce * ballRadius
                    let alpha = (torque / momentOfInertia) * SIMD3<Float>(-frictionDir.y, 0, frictionDir.x)
                    ball.angularVelocity += alpha * dt
                } else {
                    let frictionDir = (vMag > 0) ? -simd_normalize(v) : .zero
                    let frictionForce = frictionRolling * ballMass * gravity
                    ball.velocity += (frictionForce / ballMass) * frictionDir * dt

                    let alpha = (frictionForce / ballRadius / momentOfInertia)
                        * SIMD3<Float>(-frictionDir.y, 0, frictionDir.x)
                    ball.angularVelocity += alpha * dt
                }

                // Vertical spin decay
                if w.y != 0 {
                    let decay: Float = frictionSpinDecay * (w.y > 0 ? -1 : 1)
                    ball.angularVelocity.y += decay * dt
                    if (w.y > 0 && ball.angularVelocity.y < 0) ||
                       (w.y < 0 && ball.angularVelocity.y > 0) {
                        ball.angularVelocity.y = 0
                    }
                }

                // Position & orientation
                ball.position += ball.velocity * dt
                let wMag = simd_length(ball.angularVelocity)
                if wMag > 0 {
                    let axis = ball.angularVelocity / wMag
                    let angle = wMag * dt
                    let deltaQuat = quaternionFromAxisAngle(axis, angle)
                    ball.quaternion = quaternionMultiply(deltaQuat, ball.quaternion)
                    ball.quaternion = simd_normalize(ball.quaternion)
                }

                // Collide with cushions
                if abs(ball.position.x) > tableWidth - ballRadius {
                    ball.position.x = (ball.position.x > 0)
                        ? (tableWidth - ballRadius)
                        : -(tableWidth - ballRadius)
                    ball.velocity.x = -ball.velocity.x * restitutionCushion
                    // Add spin
                    let spinChange = -ball.angularVelocity.z * 0.5
                    ball.angularVelocity.z += spinChange
                    ball.angularVelocity.y = -ball.angularVelocity.y * 0.7
                }
                if abs(ball.position.y) > tableLength - ballRadius {
                    ball.position.y = (ball.position.y > 0)
                        ? (tableLength - ballRadius)
                        : -(tableLength - ballRadius)
                    ball.velocity.y = -ball.velocity.y * restitutionCushion
                    let spinChange = ball.angularVelocity.x * 0.5
                    ball.angularVelocity.x += spinChange
                    ball.angularVelocity.y = -ball.angularVelocity.y * 0.7
                }

                // Check pockets
                if checkPocket(pos: ball.position) {
                    // Mark pocketed
                    ball.velocity = SIMD2<Float>(.infinity, .infinity)
                    ball.angularVelocity = .zero
                    ball.position = .zero
                }

                balls[i] = ball
            }

            // Ball-ball collisions
            for i in 0..<15 {
                for j in (i+1)..<16 {
                    var ball1 = balls[i]
                    var ball2 = balls[j]
                    if ball1.velocity.x.isInfinite || ball2.velocity.x.isInfinite { continue }

                    let delta = ball2.position - ball1.position
                    let dist = simd_length(delta)
                    if dist < 2.0 * ballRadius && dist > 0 {
                        let normal = delta / dist
                        let relativeVel = ball1.velocity - ball2.velocity
                        let impulse = simd_dot(relativeVel, normal)
                        if impulse > 0 {
                            let impulseMag = -(1.0 + restitutionBall) * impulse / (2.0 / ballMass)
                            let impulseVector = normal * impulseMag
                            ball1.velocity += impulseVector / ballMass
                            ball2.velocity -= impulseVector / ballMass

                            // Tangential friction
                            let tangent = SIMD2<Float>(-normal.y, normal.x)
                            let relVelTangent = simd_dot(relativeVel, tangent)
                            let frictionImpulse = min(ballFriction * abs(impulseMag), abs(relVelTangent) * ballMass)
                            let frictionVector = tangent * frictionImpulse * (relVelTangent > 0 ? -1 : 1)
                            ball1.velocity += frictionVector / ballMass
                            ball2.velocity -= frictionVector / ballMass

                            // Spin changes
                            let spinChange = frictionImpulse / ballRadius / momentOfInertia
                            ball1.angularVelocity += SIMD3<Float>(-tangent.y, 0, tangent.x) * spinChange
                            ball2.angularVelocity -= SIMD3<Float>(-tangent.y, 0, tangent.x) * spinChange

                            // Positional correction
                            let overlap = 2.0 * ballRadius - dist
                            let correction = normal * (overlap * 0.5)
                            ball1.position -= correction
                            ball2.position += correction
                        }
                    }
                    balls[i] = ball1
                    balls[j] = ball2
                }
            }

            // Zero out tiny velocity/spin
            for i in 0..<16 {
                let velMag = simd_length(balls[i].velocity)
                let spinMag = simd_length(balls[i].angularVelocity)
                if velMag < 0.005 && spinMag < 0.1 {
                    balls[i].velocity = .zero
                    balls[i].angularVelocity = .zero
                }
            }
        }

        // Copy updated ball data to buffer
        let ptr = ballBuffer.contents().bindMemory(to: SIMD8<Float>.self, capacity: 16)
        for i in 0..<16 {
            ptr[i] = SIMD8<Float>(
                balls[i].position.x, balls[i].position.y,
                balls[i].velocity.x, balls[i].velocity.y,
                balls[i].quaternion.x, balls[i].quaternion.y,
                balls[i].quaternion.z, balls[i].quaternion.w
            )
        }
    }

    func rotateX(_ vector: SIMD3<Float>, _ angle: Float) -> SIMD3<Float> {
        let c = cos(angle)
        let s = sin(angle)
        return SIMD3<Float>(
            vector.x,
            vector.y * c - vector.z * s,
            vector.y * s + vector.z * c
        )
    }

    func rotateY(_ vector: SIMD3<Float>, _ angle: Float) -> SIMD3<Float> {
        let c = cos(angle)
        let s = sin(angle)
        return SIMD3<Float>(
            vector.x * c + vector.z * s,
            vector.y,
            -vector.x * s + vector.z * c
        )
    }

    private func applyCueStrike() {
        // Direction vector for the cue
        var cueDir = SIMD3<Float>(0, 0, -1)
        cueDir = rotateX(cueDir, cue3DRotate.y)
        cueDir = rotateY(cueDir, -cue3DRotate.x)
        let baseSpeed: Float = 20.0
        let velocityScale = 0.3 + 1.7 * powerAtRelease

        // Spin from tip offset
        let tipOffset3D = SIMD3<Float>(cueTipOffset.x, -cueTipOffset.y, 0)
        let spinFactor: Float = 15.0 / (2.0 * ballRadius)
        let angularVelocity = simd_cross(cueDir, tipOffset3D) * spinFactor * velocityScale
        balls[0].angularVelocity = angularVelocity

        // Subtle effect of spin on direction
        let spinEffect = simd_cross(angularVelocity, SIMD3<Float>(cueDir.x, 0, cueDir.z)) * 0.5
        let adjustedDir = simd_normalize(cueDir + spinEffect)
        let adjustedDir2D = simd_normalize(SIMD2<Float>(adjustedDir.x, adjustedDir.z))
        balls[0].velocity = adjustedDir2D * baseSpeed * velocityScale

        powerAtRelease = 0.0
    }

    // MARK: - Orbit camera pass
    func encodeOrbitRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))
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

    // MARK: - Behind-ball pass
    func encodeBehindRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0, 2.0, 0)
        var cameraTarget = SIMD3<Float>(whiteBall.position.x, 0.47, whiteBall.position.y)
        let speed = simd_length(whiteBall.velocity)

        if speed < 0.01 {
            // White ball basically stationary
            let stationaryDistance: Float = 2.5
            var offset = SIMD3<Float>(0, 0, stationaryDistance)
            offset = rotateY(offset, -cue3DRotate.x)
            cameraPosition = cameraTarget + offset
            cameraPosition.y = 0.47 + 0.7
        } else {
            // Move the camera behind the traveling ball
            let forward = simd_normalize(SIMD3<Float>(whiteBall.velocity.x, 0, whiteBall.velocity.y))
            cameraPosition = cameraTarget - (forward * 3.0)
            cameraPosition.y += 1.0
        }

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

    // MARK: - Third-person pass
    func encodeThirdRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0, 2.0, 0)
        var cameraTarget = SIMD3<Float>(whiteBall.position.x, 0.47, whiteBall.position.y)
        let speed = simd_length(whiteBall.velocity)

        if speed < 0.01 {
            let stationaryDistance: Float = 7.0
            var offset = SIMD3<Float>(0, 0, stationaryDistance)
            offset = rotateX(offset, cue3DRotate.y)
            offset = rotateY(offset, -cue3DRotate.x)
            cameraPosition = cameraTarget + offset

            let cueBaseHeight: Float = 0.47
            let verticalAdjustment = sin(cue3DRotate.y) * stationaryDistance
            cameraPosition.y = cueBaseHeight + verticalAdjustment + 0.7
        } else {
            let forward = simd_normalize(SIMD3<Float>(whiteBall.velocity.x, 0, whiteBall.velocity.y))
            cameraPosition = cameraTarget - (forward * 8.0)
            cameraPosition.y += 1.0
        }

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
    @State private var viewSizeBehind: CGSize = .zero
    @State private var initialTouchBehind: CGPoint? = nil
    @State private var initialTipOffsetBehind: SIMD2<Float> = .zero

    @State private var viewSizeThird: CGSize = .zero
    @State private var initialTouchThird: CGPoint? = nil
    @State private var initialCueYawThird: Float = 0.0
    @State private var initialCuePitchThird: Float = 0.0

    var body: some View {
        ZStack {
            // Orbiting camera (main)
            OrbitingMetalView(simulation: simulation)
                .edgesIgnoringSafeArea(.all)
                .overlay(Text("Orbiting Camera").foregroundColor(.white).padding(), alignment: .top)
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in simulation.isTouching = true }
                        .onEnded { _ in simulation.isTouching = false }
                )

            // Two smaller "picture-in-picture" views
            VStack {
                Spacer()
                HStack {
                    // Behind-ball view
                    BehindBallMetalView(simulation: simulation)
                        .frame(width: UIScreen.main.bounds.width / 2,
                               height: UIScreen.main.bounds.width / 2)
                        .background(
                            GeometryReader { geo in
                                Color.clear
                                    .onAppear { viewSizeBehind = geo.size }
                                    .onChange(of: geo.size) { newSize in viewSizeBehind = newSize }
                            }
                        )
                        .overlay(Text("Behind-Ball Camera").foregroundColor(.white).padding(),
                                 alignment: .top)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    if simulation.showCueValue == 1 && !simulation.shooting {
                                        if initialTouchBehind == nil {
                                            initialTouchBehind = value.startLocation
                                            initialTipOffsetBehind = simulation.cueTipOffset
                                        }
                                        guard let start = initialTouchBehind else { return }
                                        let dx = Float(value.location.x - start.x)
                                        let dy = Float(value.location.y - start.y)

                                        // Scale factor for offset
                                        let scaleFactor = simulation.maxTipOffset
                                            / Float(viewSizeBehind.height) * 2.0
                                        let aspect = Float(viewSizeBehind.width / viewSizeBehind.height)
                                        var newOffset = initialTipOffsetBehind + SIMD2<Float>(
                                            dx * scaleFactor * aspect,
                                            -dy * scaleFactor
                                        )
                                        let offsetLen = simd_length(newOffset)
                                        if offsetLen > simulation.maxTipOffset {
                                            newOffset *= simulation.maxTipOffset / offsetLen
                                        }
                                        let oldTip = simulation.cueTipOffset
                                        simulation.cueTipOffset = newOffset

                                        // If collision, revert
                                        if simulation.isCueColliding(
                                            offset: simulation.cueOffset,
                                            tipOffset: simulation.cueTipOffset,
                                            yaw: simulation.cue3DRotate.x,
                                            pitch: simulation.cue3DRotate.y
                                        ) {
                                            simulation.cueTipOffset = oldTip
                                        }
                                    }
                                }
                                .onEnded { _ in
                                    initialTouchBehind = nil
                                    initialTipOffsetBehind = .zero
                                }
                        )

                    // Third-person camera
                    ThirdBallMetalView(simulation: simulation)
                        .frame(width: UIScreen.main.bounds.width / 2,
                               height: UIScreen.main.bounds.width / 2)
                        .background(
                            GeometryReader { geo in
                                Color.clear
                                    .onAppear { viewSizeThird = geo.size }
                                    .onChange(of: geo.size) { newSize in viewSizeThird = newSize }
                            }
                        )
                        .overlay(Text("Third-Ball Camera (3D Cue Rotation)").foregroundColor(.white).padding(),
                                 alignment: .top)
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
                                        let dx = Float(value.location.x - start.x)
                                        let dy = Float(value.location.y - start.y)
                                        let sensitivity: Float = 0.01
                                        let newYaw = initialCueYawThird - dx * sensitivity
                                        let newPitch = initialCuePitchThird - dy * sensitivity
                                        let clampedPitch = max(-0.8, min(0.8, newPitch))

                                        let oldYaw = simulation.cue3DRotate.x
                                        let oldPitch = simulation.cue3DRotate.y
                                        simulation.cue3DRotate = SIMD2<Float>(newYaw, clampedPitch)

                                        // If collision, revert
                                        if simulation.isCueColliding(
                                            offset: simulation.cueOffset,
                                            tipOffset: simulation.cueTipOffset,
                                            yaw: simulation.cue3DRotate.x,
                                            pitch: simulation.cue3DRotate.y
                                        ) {
                                            simulation.cue3DRotate = SIMD2<Float>(oldYaw, oldPitch)
                                        }
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
            // Animation loop at 60fps
            let timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { _ in
                simulation.time += 1.0 / 60.0
                simulation.updatePhysics(deltaTime: 1.0 / 60.0)
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
