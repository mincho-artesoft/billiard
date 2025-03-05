import SwiftUI
import MetalKit
import simd

// MARK: - Metal Shaders (same code as in your snippet)
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

// Utility functions:
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

// Detect ball-ray hit:
void ballHit(float3 ro, float3 rd, thread float &dist, thread float3 &normal, thread int &id, constant Ball* balls [[buffer(2)]]) {
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

// The main scene function: returns color for each ray
float3 showScene(float3 ro, float3 rd,
                 float time, float cueOffset,
                 constant Ball* balls [[buffer(2)]],
                 int cueVisible) {
    const float hbLen = 8.0;
    const float bWid  = 0.4;
    const float2 hIn  = float2(hbLen, hbLen * 1.75) - bWid;
    const float PI    = 3.14159;

    float3 col = float3(0.05, 0.05, 0.1);
    float t = 0.0;
    const float maxDist = 50.0;

    // Ball intersection:
    float dstBall;
    float3 ballNormal;
    int ballId;
    ballHit(ro, rd, dstBall, ballNormal, ballId, balls);

    // Table/cue intersection:
    float dstTable = maxDist;
    float dstCue   = maxDist;
    float3 cueHitPos;

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

        float3 pc = p - float3(balls[0].position.x, 0.05, balls[0].position.y);
        pc.y -= 0.05;
        float aCue = sin(time * 0.5) * 0.1;
        pc.xz = -rot2D(pc.xz, aCue);
        float cueLength = 2.5;
        float ballRadius = 0.47;
        float tipOffset = cueLength;
        float maxCueOffset = -ballRadius;
        pc.z -= (maxCueOffset - cueOffset - tipOffset);
        float dCueStick = prRoundCylDf(pc, 0.1 - (0.015 / 2.5) * (pc.z + tipOffset), 0.05, cueLength);
        if (cueVisible == 0) dCueStick = 99999.0;

        float d = min(dTable, dCueStick);
        if (d < 0.0005 || t > dstBall) {
            if (dTable < dCueStick) dstTable = t;
            else { dstCue = t; cueHitPos = pc; }
            break;
        }
        t += d * 0.7;
        if (t > maxDist) break;
    }

    float3 lightPos = float3(0.0, 3.0 * hbLen, 0.0);
    float minDist = min(min(dstBall, dstTable), dstCue);

    if (minDist < maxDist) {
        if (dstBall <= min(dstTable, dstCue)) {
            // Ball shading
            float3 p = ro + rd * dstBall;
            float3 n = ballNormal;
            int id = ballId;
            if (id == 0) {
                // White ball
                col = float3(1.0);
            } else {
                // Colored / striped balls
                float c = float(id - 1);
                float3 baseColor;
                bool isStriped = (id >= 9);
                if (id == 8) {
                    baseColor = float3(0.0);  // black ball
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
            // Cue stick shading
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
            // Table shading
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

// ------------------------------------------------
// Vertex Shader (same for both pipelines)
// ------------------------------------------------
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

// ------------------------------------------------
// Orbiting-camera fragmentShader
// ------------------------------------------------
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant float2 &resolution [[buffer(0)]],
                               constant float &time        [[buffer(1)]],
                               constant Ball* balls        [[buffer(2)]],
                               constant float &cueOffset   [[buffer(3)]],
                               constant int &cueVisible    [[buffer(4)]]) {

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

    float3 col = showScene(ro, rd, time, cueOffset, balls, cueVisible);
    return float4(col, 1.0);
}

// ------------------------------------------------
// Behind-ball fragmentShader
// ------------------------------------------------
fragment float4 behindBallFragmentShader(VertexOut in [[stage_in]],
                                         constant float2 &resolution          [[buffer(0)]],
                                         constant float3 &cameraPosition      [[buffer(1)]],
                                         constant float3 &cameraTarget        [[buffer(2)]],
                                         constant Ball*  balls                [[buffer(3)]],
                                         constant float  &cueOffset           [[buffer(4)]],
                                         constant int    &cueVisible          [[buffer(5)]]) {
    float2 uv = 2.0 * in.uv - 1.0;
    uv.x *= resolution.x / resolution.y;

    float3 ro = cameraPosition;
    float3 target = cameraTarget;

    float3 vd = normalize(target - ro);
    float3 right = normalize(cross(float3(0.0, 1.0, 0.0), vd));
    float3 up = normalize(cross(vd, right));

    const float fov = 0.8;
    float3 rd = normalize(vd + right * uv.x * fov + up * uv.y * fov);

    // Reuse the same scene function, but no real 'time' needed for behind-ball
    float timeDummy = 0.0;
    float3 col = showScene(ro, rd, timeDummy, cueOffset, balls, cueVisible);
    return float4(col, 1.0);
}
"""

// MARK: - Swift Data Structures and Utilities
struct BallData {
    var position: SIMD2<Float>
    var velocity: SIMD2<Float>
    var quaternion: SIMD4<Float>
}

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

// MARK: - The Single Shared Simulation
final class BilliardSimulation: ObservableObject {
    // MARK: Metal objects
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let orbitPipeline: MTLRenderPipelineState
    private let behindPipeline: MTLRenderPipelineState

    // Shared data
    var time: Float = 0.0
    var isTouching: Bool = false  // Cue pulling
    var cueOffset: Float = 0.0
    var showCueValue: Int32 = 1   // 1 => show, 0 => hide

    // Balls
    var balls: [BallData]
    private var ballBuffer: MTLBuffer

    // Constants
    private let ballRadius: Float = 0.47
    private let tableWidth: Float = 7.6
    private let tableLength: Float = 13.3
    private let pocketRadius: Float = 0.53

    private let cuePullSpeed: Float = 1.0
    private let cueStrikeSpeed: Float = 5.0
    private let maxCueOffset: Float = 2.0

    private var hitTriggered: Bool = false
    private var shooting: Bool = false
    private var powerAtRelease: Float = 0.0

    private let friction: Float = 0.98
    private let restitution: Float = 0.9
    private let minVelocity: Float = 0.01
    private let ballMass: Float = 1.0
    private let momentOfInertia: Float = 0.4 * 0.47 * 0.47

    // Buffers used by the fragment shaders
    private var resolution = SIMD2<Float>(0,0)
    private var orbitUniformsBuffer: MTLBuffer      // [ time ]
    private var behindCamPosBuffer: MTLBuffer       // [ camera pos as float3 ]
    private var behindCamTargetBuffer: MTLBuffer    // [ camera target as float3 ]
    private var cueOffsetBuffer: MTLBuffer          // [ float ]
    private var showCueBuffer: MTLBuffer            // [ int ]

    init?() {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let cq = dev.makeCommandQueue() else {
            return nil
        }
        device = dev
        commandQueue = cq

        // Compile shaders
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: metalShader, options: nil)
        } catch {
            print("Failed to create Metal library: \(error)")
            return nil
        }
        guard
            let vertexFunction = library.makeFunction(name: "vertexShader"),
            let orbitFrag = library.makeFunction(name: "fragmentShader"),
            let behindFrag = library.makeFunction(name: "behindBallFragmentShader")
        else {
            print("Missing required shader functions.")
            return nil
        }

        do {
            // Orbit pipeline
            let orbitDescriptor = MTLRenderPipelineDescriptor()
            orbitDescriptor.vertexFunction = vertexFunction
            orbitDescriptor.fragmentFunction = orbitFrag
            orbitDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            orbitPipeline = try device.makeRenderPipelineState(descriptor: orbitDescriptor)

            // Behind pipeline
            let behindDescriptor = MTLRenderPipelineDescriptor()
            behindDescriptor.vertexFunction = vertexFunction
            behindDescriptor.fragmentFunction = behindFrag
            behindDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            behindPipeline = try device.makeRenderPipelineState(descriptor: behindDescriptor)
        } catch {
            print("Failed to create pipeline state: \(error)")
            return nil
        }

        // Prepare ball data
        let identityQuat = SIMD4<Float>(0, 0, 0, 1)
        self.balls = [
            // White ball at top
            BallData(position: SIMD2<Float>( 0.0,  5.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),

            // Just a sample arrangement
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

            // One more ball
            BallData(position: SIMD2<Float>( 0.0,  0.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat)
        ]

        // Create the buffer for these 16 balls
        var ballShaderData = [SIMD8<Float>](repeating: SIMD8<Float>(0,0,0,0,0,0,0,0), count: 16)
        for i in 0..<16 {
            ballShaderData[i] = SIMD8<Float>(
                balls[i].position.x,  balls[i].position.y,
                balls[i].velocity.x,  balls[i].velocity.y,
                balls[i].quaternion.x,balls[i].quaternion.y,
                balls[i].quaternion.z,balls[i].quaternion.w
            )
        }
        self.ballBuffer = device.makeBuffer(bytes: ballShaderData,
                                            length: MemoryLayout<SIMD8<Float>>.stride * 16,
                                            options: .storageModeShared)!

        // Create small buffers for uniforms
        orbitUniformsBuffer     = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        behindCamPosBuffer      = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        behindCamTargetBuffer   = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        cueOffsetBuffer         = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        showCueBuffer           = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared)!

        // Done
    }

    // MARK: - Billiard Physics
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

    func updatePhysics(deltaTime: Float) {
        // Cue stick control
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
            if cueOffset <= 0.0 {
                cueOffset = 0.0
                // Fire the white ball
                let baseVelocity: Float = -30.0
                let velocityScale = 0.5 + 1.5 * powerAtRelease
                balls[0].velocity = SIMD2<Float>(0.0, baseVelocity * velocityScale)
                hitTriggered = true
                shooting = false
                powerAtRelease = 0.0
                showCueValue = 0
            }
        }

        // Wait for all balls to stop => restore cue
        if !isTouching && hitTriggered {
            var allStopped = true
            for ball in balls {
                if ball.velocity.x.isInfinite { continue } // pocketed
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

        // Sub-steps
        let subSteps = 2
        let dt = deltaTime / Float(subSteps)

        for _ in 0..<subSteps {
            // 1) Integrate
            for i in 0..<16 {
                var pos = balls[i].position
                var vel = balls[i].velocity
                var quat = balls[i].quaternion

                if vel.x.isInfinite { continue } // pocketed

                // Position
                pos += vel * dt

                // Spin
                let angVel3D = SIMD3<Float>(0.0, vel.x / ballRadius, -vel.y / ballRadius)
                let angSpeed = simd_length(angVel3D)
                if angSpeed > 0.0 {
                    let axis = angVel3D / angSpeed
                    let dq = quaternionFromAxisAngle(axis, angSpeed * dt)
                    quat = quaternionMultiply(dq, quat)
                    quat = simd_normalize(quat)
                }

                // Check pockets
                if checkPocket(pos: pos) {
                    vel = SIMD2<Float>(.infinity, .infinity)
                    pos = SIMD2<Float>(0.0, 0.0) // effectively "gone"
                } else {
                    // Collide with table edges
                    if abs(pos.x) > tableWidth - ballRadius {
                        pos.x = (pos.x > 0)
                            ?  tableWidth - ballRadius
                            : -tableWidth + ballRadius
                        vel.x = -vel.x * restitution
                        let angVel = vel.y / ballRadius
                        let dq = quaternionFromAxisAngle(SIMD3<Float>(0,0,1), angVel*dt)
                        quat = quaternionMultiply(dq, quat)
                        quat = simd_normalize(quat)
                    }
                    if abs(pos.y) > tableLength - ballRadius {
                        pos.y = (pos.y > 0)
                            ?  tableLength - ballRadius
                            : -tableLength + ballRadius
                        vel.y = -vel.y * restitution
                        let angVel = -vel.x / ballRadius
                        let dq = quaternionFromAxisAngle(SIMD3<Float>(0,1,0), angVel*dt)
                        quat = quaternionMultiply(dq, quat)
                        quat = simd_normalize(quat)
                    }
                }

                // Friction
                vel *= friction
                if simd_length(vel) < minVelocity {
                    vel = .zero
                }

                balls[i].position   = pos
                balls[i].velocity   = vel
                balls[i].quaternion = quat
            }

            // 2) Ball-ball collisions
            for i in 0..<15 {
                for j in (i+1)..<16 {
                    var pos1 = balls[i].position
                    var pos2 = balls[j].position
                    var vel1 = balls[i].velocity
                    var vel2 = balls[j].velocity
                    var quat1 = balls[i].quaternion
                    var quat2 = balls[j].quaternion

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

                            // Basic friction
                            let tangent = SIMD2<Float>(-normal.y, normal.x)
                            let relVelTangent = simd_dot(relativeVel, tangent)
                            let frictionImpulse = relVelTangent * 0.2

                            let angVel1 = frictionImpulse / momentOfInertia
                            let angVel2 = -frictionImpulse / momentOfInertia

                            let axis1 = SIMD3<Float>(tangent.x, tangent.y, 0)
                            let axis2 = -axis1
                            let dq1 = quaternionFromAxisAngle(axis1, angVel1 * dt)
                            let dq2 = quaternionFromAxisAngle(axis2, angVel2 * dt)

                            quat1 = quaternionMultiply(dq1, quat1)
                            quat2 = quaternionMultiply(dq2, quat2)
                            quat1 = simd_normalize(quat1)
                            quat2 = simd_normalize(quat2)

                            // Positional correction
                            let overlap = (2.0 * ballRadius) - dist
                            let correction = normal * (overlap * 0.5)
                            pos1 -= correction
                            pos2 += correction
                        }

                        balls[i].position   = pos1
                        balls[j].position   = pos2
                        balls[i].velocity   = vel1
                        balls[j].velocity   = vel2
                        balls[i].quaternion = quat1
                        balls[j].quaternion = quat2
                    }
                }
            }
        }

        // 3) Write back final ball data to GPU
        let ptr = ballBuffer.contents().bindMemory(to: SIMD8<Float>.self, capacity: 16)
        for i in 0..<16 {
            ptr[i] = SIMD8<Float>(
                balls[i].position.x,  balls[i].position.y,
                balls[i].velocity.x,  balls[i].velocity.y,
                balls[i].quaternion.x,balls[i].quaternion.y,
                balls[i].quaternion.z,balls[i].quaternion.w
            )
        }
    }

    // MARK: - Encoding the Render Passes
    func encodeOrbitRenderPass(encoder: MTLRenderCommandEncoder,
                               viewSize: CGSize)
    {
        self.resolution = SIMD2<Float>(Float(viewSize.width),
                                       Float(viewSize.height))

        // Pass data to orbit fragment shader
        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.size, index: 0)

        // time
        let timePtr = orbitUniformsBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        timePtr[0] = self.time
        encoder.setFragmentBuffer(orbitUniformsBuffer, offset: 0, index: 1)

        // ball array
        encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 2)

        // cue offset
        let cuePtr = cueOffsetBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        cuePtr[0] = self.cueOffset
        encoder.setFragmentBuffer(cueOffsetBuffer, offset: 0, index: 3)

        // show cue
        let showPtr = showCueBuffer.contents().bindMemory(to: Int32.self, capacity: 1)
        showPtr[0] = self.showCueValue
        encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 4)

        encoder.setRenderPipelineState(orbitPipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    func encodeBehindRenderPass(encoder: MTLRenderCommandEncoder,
                                viewSize: CGSize)
    {
        self.resolution = SIMD2<Float>(Float(viewSize.width),
                                       Float(viewSize.height))

        // We'll position camera behind the white ball
        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0, 2.0, 0)
        var cameraTarget   = SIMD3<Float>(0, 0.0, 0)
        let speed = simd_length(whiteBall.velocity)
        if speed < 0.01 {
            // Stationary => put the camera behind the ball at some distance
            cameraTarget = SIMD3<Float>(whiteBall.position.x, 0.05, whiteBall.position.y)
            let zForward: Float = 2.5
            cameraPosition = cameraTarget + SIMD3<Float>(0, 0.7, zForward)
        } else {
            // Ball is moving => behind the ball in direction of velocity
            let forward = simd_normalize(SIMD3<Float>(whiteBall.velocity.x, 0, whiteBall.velocity.y) + 0.000001)
            cameraTarget = SIMD3<Float>(whiteBall.position.x, 0.05, whiteBall.position.y)
            cameraPosition = cameraTarget - (forward * 3.0)
            cameraPosition.y += 1.0
        }

        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.size, index: 0)

        // camera pos
        let camPosPtr = behindCamPosBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)
        camPosPtr[0] = cameraPosition
        encoder.setFragmentBuffer(behindCamPosBuffer, offset: 0, index: 1)

        // camera target
        let camTgtPtr = behindCamTargetBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)
        camTgtPtr[0] = cameraTarget
        encoder.setFragmentBuffer(behindCamTargetBuffer, offset: 0, index: 2)

        // ball array
        encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 3)

        // cue offset
        let cuePtr = cueOffsetBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        cuePtr[0] = self.cueOffset
        encoder.setFragmentBuffer(cueOffsetBuffer, offset: 0, index: 4)

        // show cue
        let showPtr = showCueBuffer.contents().bindMemory(to: Int32.self, capacity: 1)
        showPtr[0] = self.showCueValue
        encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 5)

        encoder.setRenderPipelineState(behindPipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
}

// MARK: - Orbiting Camera View
struct OrbitingMetalView: UIViewRepresentable {
    @ObservedObject var simulation: BilliardSimulation

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: OrbitingMetalView
        init(_ parent: OrbitingMetalView) {
            self.parent = parent
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let commandBuffer = parent.simulation.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
            else { return }

            // Encode orbit pass
            parent.simulation.encodeOrbitRenderPass(encoder: encoder, viewSize: view.drawableSize)

            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: simulation.device)
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        // No-op: we do everything in draw(in:) plus the external physics timer
    }
}

// MARK: - Behind-Ball Camera View
struct BehindBallMetalView: UIViewRepresentable {
    @ObservedObject var simulation: BilliardSimulation

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: BehindBallMetalView
        init(_ parent: BehindBallMetalView) {
            self.parent = parent
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let commandBuffer = parent.simulation.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
            else { return }

            // Encode behind-ball pass
            parent.simulation.encodeBehindRenderPass(encoder: encoder, viewSize: view.drawableSize)

            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: simulation.device)
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        // No-op
    }
}

// MARK: - The SwiftUI ContentView
struct ContentView: View {
    @StateObject private var simulation = BilliardSimulation()!

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            // Main Orbiting Camera View
            OrbitingMetalView(simulation: simulation)
                .edgesIgnoringSafeArea(.all)
                .overlay(
                    Text("Orbiting Camera")
                        .foregroundColor(.white)
                        .padding(),
                    alignment: .top
                )
                // <-- Attach the cue-stick gesture here, on the orbiting view only:
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in
                            simulation.isTouching = true
                        }
                        .onEnded { _ in
                            simulation.isTouching = false
                        }
                )

            // Smaller behind-ball camera in bottom-left corner
            BehindBallMetalView(simulation: simulation)
                .frame(width: UIScreen.main.bounds.width / 2,
                       height: UIScreen.main.bounds.width / 2)
                .overlay(
                    Text("Behind-Ball Camera")
                        .foregroundColor(.white)
                        .padding(),
                    alignment: .top
                )
        }
        // Remove the .gesture(...) here
        .onAppear {
            let timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0,
                                             repeats: true) { _ in
                simulation.time += 1.0/60.0
                simulation.updatePhysics(deltaTime: 1.0/60.0)
            }
            RunLoop.current.add(timer, forMode: .common)
        }
    }
}

// MARK: - SwiftUI Preview
#Preview {
    ContentView()
}
