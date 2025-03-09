import SwiftUI
import MetalKit
import simd

// MARK: - Metal Shader Source
let metalShader = """
#include <metal_stdlib>
using namespace metal;

// MARK: - Noise Functions for Procedural Texture
float hash(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f); // Smoothstep
    float a = hash(i + float2(0.0, 0.0));
    float b = hash(i + float2(1.0, 0.0));
    float c = hash(i + float2(0.0, 1.0));
    float d = hash(i + float2(1.0, 1.0));
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(float2 p, int octaves) {
    float v = 0.0;
    float a = 0.5;
    float2 shift = float2(100.0);
    for (int i = 0; i < octaves; ++i) {
        v += a * noise(p);
        p = p * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

// MARK: - Existing Utility Functions
struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Ball {
    float2 position;
    float2 velocity;
    float4 quaternion;
    float height;
};

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

void ballHit(float3 ro, float3 rd, thread float &dist, thread float3 &normal,
             thread int &id, constant Ball* balls [[buffer(2)]]) {
    const int nBall = 16;
    const float rad = 0.47;
    dist = 50.0;
    normal = float3(0.0);
    id = -1;
    for (int n = 0; n < nBall; n++) {
        float3 ballPos = float3(balls[n].position.x, balls[n].height, balls[n].position.y);
        float3 u = ro - ballPos;
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

    float dstBall;
    float3 ballNormal;
    int ballId;
    ballHit(ro, rd, dstBall, ballNormal, ballId, balls);

    float dstTable = maxDist;
    float dstCue = maxDist;
    float3 cueHitPos;

    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * t;

        float dSurface = prBoxDf(p, float3(hIn.x, 0.4, hIn.y));
        float3 pb = p; 
        pb.y -= -0.6;
        float dBorder = prRoundBoxDf(pb, float3(hIn.x + 0.6, 0.5, hIn.y + 0.6), 0.2);
        float dTable = max(dBorder, -dSurface);

        float2 hInPocket = hIn - bWid + 0.03;
        float3 q = p;
        q.x = abs(q.x) - hInPocket.x;
        q.z = fmod(q.z + 0.5 * hInPocket.y, hInPocket.y) - 0.5 * hInPocket.y;
        float pocketDist = length(q.xz);
        dTable = smoothMax(dTable, 0.53 - pocketDist, 0.01);

        float3 pc = p - float3(balls[0].position.x, balls[0].height, balls[0].position.y);
        pc.y -= 0.01;

        float baseAngle = sin(time * 0.5) * 0.1;
        pc = rotateX(pc, cue3DRotate.y);
        float finalYaw = baseAngle + cue3DRotate.x;
        pc = rotateY(pc, finalYaw);

        float cueLength = 2.5;
        float ballRadius = 0.47;
        float tipOffset = cueLength;
        float maxCueOffset = -ballRadius;
        pc.z += (maxCueOffset - cueOffset - tipOffset);
        pc = rotateY(pc, 3.14159);

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

            // Procedural Felt Texture with Enhanced Color Variation and Shadows
            float2 feltUV = p.xz * 0.5; // Scale for texture detail
            float feltNoise = fbm(feltUV, 4); // Base noise for texture
            float fiberDetail = noise(feltUV * 15.0); // Higher frequency for fine fibers
            float shadowNoise = fbm(feltUV * 0.2, 3); // Low-frequency noise for shadows

            // Base felt color (green)
            float3 feltBaseColor = float3(0.1, 0.5, 0.2);
            // Darker fibrous strands
            float3 fiberColor = float3(0.05, 0.3, 0.1);
            // Mix base and fiber colors based on noise
            float fiberMix = smoothstep(0.6, 0.8, fiberDetail);
            float3 feltColor = mix(feltBaseColor, fiberColor, fiberMix);
            // Additional color variation with base noise
            feltColor *= (0.7 + 0.3 * feltNoise);

            // Perturb normal for fibrous bumpiness
            float3 feltNormal = n;
            float noiseGradX = fbm(feltUV + float2(0.01, 0.0), 4) - feltNoise;
            float noiseGradZ = fbm(feltUV + float2(0.0, 0.01), 4) - feltNoise;
            feltNormal += float3(noiseGradX, 0.0, noiseGradZ) * 0.07; // Slightly increased perturbation
            feltNormal = normalize(feltNormal);

            // Shadow effect
            float shadowFactor = smoothstep(0.3, 0.7, shadowNoise);
            float shadowStrength = 0.4; // Adjust shadow intensity
            float ambient = 0.3; // Minimum lighting level

            float2 pocketCheck = float2(
                abs(p.x) - (hIn.x - bWid + 0.03),
                fmod(p.z + 0.5 * (hIn.y - bWid + 0.03), (hIn.y - bWid + 0.03)) - 0.5 * (hIn.y - bWid + 0.03)
            );
            float pocketDist = length(pocketCheck);
            if (pocketDist < 0.53) {
                col = float3(0.0); // Pocket color
            } else if (max(abs(p.x) - hIn.x, abs(p.z) - hIn.y) < 0.3) {
                col = float3(0.1, 0.5, 0.3); // Cushion color
            } else {
                col = feltColor; // Apply procedural felt texture
            }

            float diff = max(dot(feltNormal, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, feltNormal);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            // Apply lighting with shadow
            col *= (ambient + (1.0 - ambient) * diff * (1.0 - shadowStrength * (1.0 - shadowFactor)));
            col += float3(0.15) * spec * (0.5 + 0.5 * feltNoise); // Reduced specular intensity
        }
    }

    return clamp(col, 0.0, 1.0);
}

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
    var height: Float = 0.01
    var verticalVelocity: Float = 0.0
    var angularVelocity: SIMD3<Float>
    var quaternion: SIMD4<Float>
}

// MARK: - Shader-Compatible Ball Data
struct BallShaderData {
    var position: SIMD2<Float>
    var velocity: SIMD2<Float>
    var quaternion: SIMD4<Float>
    var height: Float
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
    private let frictionKinetic: Float = 0.25    // Increased for quicker sliding stop
    private let frictionRolling: Float = 0.015   // Increased for realistic rolling decay
    private let frictionSpinDecay: Float = 8.0   // Increased for faster spin decay
    private let restitutionBall: Float = 0.95
    private let restitutionCushion: Float = 0.8
    private let ballFriction: Float = 0.05

    private var hitTriggered: Bool = false
    private var powerAtRelease: Float = 0.0

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

        var ballShaderData = [BallShaderData](repeating: BallShaderData(position: .zero, velocity: .zero, quaternion: .zero, height: 0.0), count: 16)
        for i in 0..<16 {
            ballShaderData[i] = BallShaderData(
                position: balls[i].position,
                velocity: balls[i].velocity,
                quaternion: balls[i].quaternion,
                height: balls[i].height
            )
        }
        self.ballBuffer = device.makeBuffer(bytes: ballShaderData,
                                            length: MemoryLayout<BallShaderData>.stride * 16,
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

    private func checkPocket(pos: SIMD2<Float>, height: Float) -> Bool {
        let pocketPositions: [SIMD2<Float>] = [
            SIMD2<Float>(-tableWidth + ballRadius, -tableLength + ballRadius), // Bottom-left corner
            SIMD2<Float>( tableWidth - ballRadius, -tableLength + ballRadius), // Bottom-right corner
            SIMD2<Float>(-tableWidth + ballRadius, 0.0),                      // Middle-left
            SIMD2<Float>( tableWidth - ballRadius, 0.0),                      // Middle-right
            SIMD2<Float>(-tableWidth + ballRadius,  tableLength - ballRadius), // Top-left corner
            SIMD2<Float>( tableWidth - ballRadius,  tableLength - ballRadius), // Top-right corner
        ]
        for p in pocketPositions {
            if simd_length(pos - p) < pocketRadius && height <= 0.01 + ballRadius {
                return true
            }
        }
        return false
    }

    func updatePhysics(deltaTime: Float) {
        if isTouching && !shooting {
            cueOffset += cuePullSpeed * deltaTime
            if cueOffset > maxCueOffset { cueOffset = maxCueOffset }
        } else if !isTouching && cueOffset > 0.0 {
            if !shooting {
                powerAtRelease = cueOffset / maxCueOffset
                shooting = true
            }
            cueOffset -= cueStrikeSpeed * deltaTime
            if (cueOffset <= 0.0) {
                cueOffset = 0.0
                applyCueStrike()
                shooting = false
                hitTriggered = true
                showCueValue = 0
            }
        }

        if !isTouching && hitTriggered {
            var allStopped = true
            for ball in balls {
                if simd_length(ball.velocity) > 0.01 || abs(ball.verticalVelocity) > 0.01 || simd_length(ball.angularVelocity) > 0.05 {
                    allStopped = false
                    break
                }
            }
            if allStopped {
                hitTriggered = false
                showCueValue = 1
            }
        }

        let subSteps = 8
        let dt = deltaTime / Float(subSteps)

        for _ in 0..<subSteps {
            for i in 0..<16 {
                var ball = balls[i]
                if ball.velocity.x.isInfinite { continue }

                let v = ball.velocity
                let w = ball.angularVelocity
                let vMag = simd_length(v)

                // Gravity and vertical motion
                ball.verticalVelocity -= gravity * dt
                ball.height += ball.verticalVelocity * dt
                if ball.height <= 0.01 {
                    ball.height = 0.01
                    if ball.verticalVelocity < 0 {
                        ball.verticalVelocity = -ball.verticalVelocity * restitutionCushion
                        if abs(ball.verticalVelocity) < 0.1 { ball.verticalVelocity = 0.0 }
                    }
                }

                if ball.height <= 0.01 + 0.001 {
                    let relativeVelocityAtContact = v - ballRadius * SIMD2<Float>(-w.z, w.x)
                    let sliding = simd_length(relativeVelocityAtContact) > 0.02

                    if sliding {
                        let frictionDir = -simd_normalize(relativeVelocityAtContact)
                        let frictionForce = frictionKinetic * ballMass * gravity
                        let accel = frictionForce / ballMass * frictionDir
                        ball.velocity += accel * dt
                        let torque = frictionForce * ballRadius
                        let alpha = torque / momentOfInertia * SIMD3<Float>(-frictionDir.y, 0, frictionDir.x)
                        ball.angularVelocity += alpha * dt
                    } else if vMag > 0 {
                        let frictionDir = -simd_normalize(v)
                        let frictionForce = frictionRolling * ballMass * gravity
                        let accel = frictionForce / ballMass * frictionDir
                        ball.velocity += accel * dt
                        let alpha = frictionForce / ballRadius / momentOfInertia
                            * SIMD3<Float>(-frictionDir.y, 0, frictionDir.x)
                        ball.angularVelocity += alpha * dt
                    }

                    // Enhanced spin decay
                    if simd_length(w) > 0 {
                        let wMag = simd_length(w)
                        let decay = -simd_normalize(w) * frictionSpinDecay * dt
                        ball.angularVelocity += decay
                        if dot(w + decay, w) <= 0 { ball.angularVelocity = .zero }
                    }
                }

                ball.position += ball.velocity * dt

                let wMag = simd_length(ball.angularVelocity)
                if wMag > 0 {
                    let axis = ball.angularVelocity / wMag
                    let angle = wMag * dt
                    let deltaQuat = quaternionFromAxisAngle(axis, angle)
                    ball.quaternion = quaternionMultiply(deltaQuat, ball.quaternion)
                    ball.quaternion = normalize(ball.quaternion)
                }

                // Cushion collisions
                if abs(ball.position.x) > tableWidth - ballRadius && ball.height <= 0.01 + ballRadius {
                    ball.position.x = (ball.position.x > 0) ? (tableWidth - ballRadius)
                                                            : -(tableWidth - ballRadius)
                    ball.velocity.x = -ball.velocity.x * restitutionCushion
                    let spinChange = -ball.angularVelocity.z * 0.5
                    ball.angularVelocity.z += spinChange
                    ball.angularVelocity.y *= 0.6
                }
                if abs(ball.position.y) > tableLength - ballRadius && ball.height <= 0.01 + ballRadius {
                    ball.position.y = (ball.position.y > 0) ? (tableLength - ballRadius)
                                                            : -(tableLength - ballRadius)
                    ball.velocity.y = -ball.velocity.y * restitutionCushion
                    let spinChange = ball.angularVelocity.x * 0.5
                    ball.angularVelocity.x += spinChange
                    ball.angularVelocity.y *= 0.6
                }

                // Pocket check
                if (i != 0 || ball.height <= 0.01 + ballRadius) && checkPocket(pos: ball.position, height: ball.height) {
                    ball.velocity = SIMD2<Float>(.infinity, .infinity)
                    ball.verticalVelocity = 0.0
                    ball.angularVelocity = .zero
                    ball.position = .zero
                    ball.height = 0.01
                }

                balls[i] = ball
            }

            // Ball collisions
            for i in 0..<15 {
                for j in (i+1)..<16 {
                    var ball1 = balls[i]
                    var ball2 = balls[j]
                    if ball1.velocity.x.isInfinite || ball2.velocity.x.isInfinite { continue }

                    let delta = ball2.position - ball1.position
                    let dist = simd_length(delta)
                    let heightDiff = abs(ball1.height - ball2.height)

                    if dist < 2.0 * ballRadius && dist > 0 && (heightDiff < ballRadius || (ball1.height <= 0.01 + ballRadius && ball2.height <= 0.01 + ballRadius)) {
                        let normal = delta / dist
                        let relativeVel = ball1.velocity - ball2.velocity
                        let impulse = simd_dot(relativeVel, normal)
                        if impulse > 0 {
                            let impulseMag = -(1.0 + restitutionBall) * impulse / (2.0 / ballMass)
                            let impulseVector = normal * impulseMag
                            ball1.velocity += impulseVector / ballMass
                            ball2.velocity -= impulseVector / ballMass

                            let tangent = SIMD2<Float>(-normal.y, normal.x)
                            let relVelTangent = simd_dot(relativeVel, tangent)
                            let frictionImpulse = min(ballFriction * abs(impulseMag),
                                                      abs(relVelTangent) * ballMass)
                            let frictionVector = tangent * frictionImpulse * (relVelTangent > 0 ? -1 : 1)
                            ball1.velocity += frictionVector / ballMass
                            ball2.velocity -= frictionVector / ballMass

                            let spinChange = frictionImpulse / ballRadius / momentOfInertia
                            ball1.angularVelocity += SIMD3<Float>(-tangent.y, 0, tangent.x) * spinChange
                            ball2.angularVelocity -= SIMD3<Float>(-tangent.y, 0, tangent.x) * spinChange

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

            // Stop small movements
            for i in 0..<16 {
                let vMag = simd_length(balls[i].velocity)
                let wMag = simd_length(balls[i].angularVelocity)
                if vMag < 0.01 && abs(balls[i].verticalVelocity) < 0.01 && wMag < 0.05 {
                    balls[i].velocity = .zero
                    balls[i].verticalVelocity = 0.0
                    balls[i].angularVelocity = .zero
                }
            }
        }

        let ptr = ballBuffer.contents().bindMemory(to: BallShaderData.self, capacity: 16)
        for i in 0..<16 {
            ptr[i] = BallShaderData(
                position: balls[i].position,
                velocity: balls[i].velocity,
                quaternion: balls[i].quaternion,
                height: balls[i].height
            )
        }
    }

    func rotateX(_ vector: SIMD3<Float>, _ angle: Float) -> SIMD3<Float> {
        let cosA = cos(angle)
        let sinA = sin(angle)
        return SIMD3<Float>(
            vector.x,
            vector.y * cosA - vector.z * sinA,
            vector.y * sinA + vector.z * cosA
        )
    }

    func rotateY(_ vector: SIMD3<Float>, _ angle: Float) -> SIMD3<Float> {
        let cosA = cos(angle)
        let sinA = sin(angle)
        return SIMD3<Float>(
            vector.x * cosA + vector.z * sinA,
            vector.y,
            -vector.x * sinA + vector.z * cosA
        )
    }

    private func applyCueStrike() {
        var cueDir = SIMD3<Float>(0, 0, -1)
        cueDir = rotateX(cueDir, cue3DRotate.y)
        cueDir = rotateY(cueDir, -cue3DRotate.x)
        let cueDir2D = normalize(SIMD2<Float>(cueDir.x, cueDir.z))

        let baseSpeed: Float = 15.0  // Reduced base speed for more realistic motion
        let velocityScale = 0.5 + 1.5 * powerAtRelease

        let tipOffset3D = SIMD3<Float>(cueTipOffset.x, -cueTipOffset.y, 0)
        let spinFactor: Float = 10.0 / (2.0 * ballRadius)  // Reduced spin factor
        let angularVelocity = cross(cueDir, tipOffset3D) * spinFactor * velocityScale
        balls[0].angularVelocity = angularVelocity

        let jumpFactor = -sin(cue3DRotate.y) * baseSpeed * velocityScale
        balls[0].verticalVelocity = jumpFactor > 0 ? jumpFactor : 0.0

        let spinEffect = cross(angularVelocity, SIMD3<Float>(cueDir.x, 0, cueDir.z)) * 0.3
        let adjustedDir = normalize(cueDir + spinEffect)
        let adjustedDir2D = normalize(SIMD2<Float>(adjustedDir.x, adjustedDir.z))
        balls[0].velocity = adjustedDir2D * baseSpeed * velocityScale

        powerAtRelease = 0.0
    }

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

    func encodeBehindRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0, 2.0, 0)
        var cameraTarget = SIMD3<Float>(whiteBall.position.x, whiteBall.height, whiteBall.position.y)
        let speed = simd_length(whiteBall.velocity)

        if speed < 0.01 {
            let stationaryDistance: Float = 2.5
            var offset = SIMD3<Float>(0, 0, stationaryDistance)
            offset = rotateY(offset, -cue3DRotate.x)
            cameraPosition = cameraTarget + offset
            cameraPosition.y = 0.7
        } else {
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

    func encodeThirdRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0, 2.0, 0)
        var cameraTarget = SIMD3<Float>(whiteBall.position.x, whiteBall.height, whiteBall.position.y)
        let speed = simd_length(whiteBall.velocity)

        if speed < 0.01 {
            let stationaryDistance: Float = 7.0
            var offset = SIMD3<Float>(0, 0, stationaryDistance)
            offset = rotateX(offset, cue3DRotate.y)
            offset = rotateY(offset, -cue3DRotate.x)
            cameraPosition = cameraTarget + offset
            let cueBaseHeight: Float = 0.01
            let cueAngleVertical = cue3DRotate.y
            let verticalAdjustment = sin(cueAngleVertical) * stationaryDistance
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
            OrbitingMetalView(simulation: simulation)
                .edgesIgnoringSafeArea(.all)
                .overlay(Text("Orbiting Camera").foregroundColor(.white).padding(), alignment: .top)
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in simulation.isTouching = true }
                        .onEnded { _ in simulation.isTouching = false }
                )

            VStack {
                Spacer()
                HStack {
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
                        .overlay(Text("Behind-Ball Camera").foregroundColor(.white).padding(), alignment: .top)
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
                                            -deltaY * scaleFactor
                                        )
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
                                        let deltaX = Float(value.location.x - start.x)
                                        let deltaY = Float(value.location.y - start.y)
                                        let sensitivity: Float = 0.01
                                        let newYaw = initialCueYawThird - deltaX * sensitivity
                                        let newPitch = initialCuePitchThird - deltaY * sensitivity
                                        let clampedPitch = max(-0.8, min(0.8, newPitch))
                                        simulation.cue3DRotate = SIMD2<Float>(newYaw, clampedPitch)
                                    }
                                }
                                .onEnded { _ in initialTouchThird = nil }
                        )
                }
                .padding(.bottom, 20)
            }
        }
        .onAppear {
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
