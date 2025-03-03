import SwiftUI
import MetalKit
import simd

// MARK: - Metal Shader (Unchanged)
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

float3 showScene(float3 ro, float3 rd, float time, float cueOffset, constant Ball* balls [[buffer(2)]], int cueVisible) {
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
    float dstCue   = maxDist;
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
                float2 uv = float2(atan2(rotatedNormal.x, rotatedNormal.z) / (2.0 * PI) + 0.5, acos(rotatedNormal.y) / PI);
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
        } else if (dstCue < dstTable) {
            float3 p = ro + rd * dstCue;
            float3 eps = float3(0.001, 0.0, 0.0);
            float3 n = normalize(float3(
                prRoundCylDf(cueHitPos + eps.xyy, 0.1, 0.05, 2.5) - prRoundCylDf(cueHitPos - eps.xyy, 0.1, 0.05, 2.5),
                prRoundCylDf(cueHitPos + eps.yxy, 0.1, 0.05, 2.5) - prRoundCylDf(cueHitPos - eps.yxy, 0.1, 0.05, 2.5),
                prRoundCylDf(cueHitPos + eps.yyx, 0.1, 0.05, 2.5) - prRoundCylDf(cueHitPos - eps.yyx, 0.1, 0.05, 2.5)
            ));
            col = (cueHitPos.z < 2.2) ? float3(0.5, 0.3, 0.0) : float3(0.7, 0.7, 0.3);
            float diff = max(dot(n, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col *= 0.3 + 0.7 * diff;
            col += float3(0.2) * spec;
        } else {
            float3 p = ro + rd * dstTable;
            float3 eps = float3(0.001, 0.0, 0.0);
            float3 n = normalize(float3(
                prBoxDf(p + eps.xyy, float3(hIn.x, 0.4, hIn.y)) - prBoxDf(p - eps.xyy, float3(hIn.x, 0.4, hIn.y)),
                prBoxDf(p + eps.yxy, float3(hIn.x, 0.4, hIn.y)) - prBoxDf(p - eps.yxy, float3(hIn.x, 0.4, hIn.y)),
                prBoxDf(p + eps.yyx, float3(hIn.x, 0.4, hIn.y)) - prBoxDf(p - eps.yyx, float3(hIn.x, 0.4, hIn.y))
            ));
            float2 pocketCheck = float2(abs(p.x) - (hIn.x - bWid + 0.03), fmod(p.z + 0.5 * (hIn.y - bWid + 0.03), (hIn.y - bWid + 0.03)) - 0.5 * (hIn.y - bWid + 0.03));
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

vertex VertexOut vertexShader(uint vertexID [[vertex_id]]) {
    constexpr float2 positions[4] = {float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0)};
    constexpr float2 uvs[4] = {float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0), float2(1.0, 1.0)};
    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]], constant float2 &resolution [[buffer(0)]], constant float &time [[buffer(1)]], constant Ball* balls [[buffer(2)]], constant float &cueOffset [[buffer(3)]], constant int &cueVisible [[buffer(4)]]) {
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
"""

// MARK: - Ball Data Structure
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
    return SIMD4<Float>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    )
}

// MARK: - Metal View
struct MetalView: UIViewRepresentable {
    @Binding var time: Float
    @Binding var isTouching: Bool

    class Coordinator: NSObject, MTKViewDelegate {
        private let parent: MetalView
        private let device: MTLDevice?
        private let commandQueue: MTLCommandQueue?
        private var pipelineState: MTLRenderPipelineState?

        private var lastFrameTime: Float
        private var balls: [BallData]
        private var ballBuffer: MTLBuffer?

        private var cueOffset: Float = 0.0
        private var cueBuffer: MTLBuffer?

        private var showCueValue: Int32 = 1
        private var showCueBuffer: MTLBuffer?

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

        init(parent: MetalView) {
            self.parent = parent
            self.device = MTLCreateSystemDefaultDevice()
            self.commandQueue = device?.makeCommandQueue()
            self.lastFrameTime = parent.time
            let identityQuat = SIMD4<Float>(0.0, 0.0, 0.0, 1.0)
            self.balls = [
                BallData(position: SIMD2<Float>( 0.0,  5.0), velocity: SIMD2<Float>(0.0, 0.0), quaternion: identityQuat),
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

            super.init()

            if let device = device {
                var ballShaderData = balls.map { SIMD8<Float>($0.position.x, $0.position.y, $0.velocity.x, $0.velocity.y, $0.quaternion.x, $0.quaternion.y, $0.quaternion.z, $0.quaternion.w) }
                ballBuffer = device.makeBuffer(bytes: &ballShaderData, length: MemoryLayout<SIMD8<Float>>.stride * 16, options: .storageModeShared)
                cueBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
                showCueBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared)

                do {
                    let library = try device.makeLibrary(source: metalShader, options: nil)
                    guard let vertexFunction = library.makeFunction(name: "vertexShader"),
                          let fragmentFunction = library.makeFunction(name: "fragmentShader") else {
                        print("Shader functions not found.")
                        return
                    }
                    let descriptor = MTLRenderPipelineDescriptor()
                    descriptor.vertexFunction = vertexFunction
                    descriptor.fragmentFunction = fragmentFunction
                    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
                    pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
                } catch {
                    print("Failed to create pipeline state: \(error)")
                }
            }
        }

        func checkPocket(pos: SIMD2<Float>) -> Bool {
            let pocketPositions: [SIMD2<Float>] = [
                SIMD2<Float>(-tableWidth + ballRadius, -tableLength + ballRadius),
                SIMD2<Float>( tableWidth - ballRadius, -tableLength + ballRadius),
                SIMD2<Float>(-tableWidth + ballRadius,  0.0),
                SIMD2<Float>( tableWidth - ballRadius,  0.0),
                SIMD2<Float>(-tableWidth + ballRadius,  tableLength - ballRadius),
                SIMD2<Float>( tableWidth - ballRadius,  tableLength - ballRadius),
            ]
            for pocket in pocketPositions {
                if length(pos - pocket) < pocketRadius {
                    return true
                }
            }
            return false
        }

        func updatePhysics(deltaTime: Float) {
            guard let ballBuffer = ballBuffer else { return }
            let ballShaderData = ballBuffer.contents().bindMemory(to: SIMD8<Float>.self, capacity: 16)

            // Cue stick control
            if parent.isTouching && !shooting {
                cueOffset += cuePullSpeed * deltaTime
                if (cueOffset > maxCueOffset) {
                    cueOffset = maxCueOffset
                }
            } else if !parent.isTouching && cueOffset > 0.0 {
                if !shooting {
                    powerAtRelease = cueOffset / maxCueOffset
                    shooting = true
                }
                cueOffset -= cueStrikeSpeed * deltaTime
                if (cueOffset <= 0.0) {
                    cueOffset = 0.0
                    let baseVelocity: Float = -30.0 // Increased from -5.0 to -15.0 for stronger hit
                    let velocityScale = 0.5 + 1.5 * powerAtRelease // Range: -7.5 to -22.5 m/s
                    balls[0].velocity = SIMD2<Float>(0.0, baseVelocity * velocityScale)
                    hitTriggered = true
                    shooting = false
                    powerAtRelease = 0.0
                    showCueValue = 0
                }
            }

            if !parent.isTouching && hitTriggered {
                var allStopped = true
                for ball in balls {
                    let vel = ball.velocity
                    if vel.x.isInfinite { continue }
                    if length(vel) > minVelocity {
                        allStopped = false
                        break
                    }
                }
                if allStopped {
                    hitTriggered = false
                    showCueValue = 1
                }
            }

            let subSteps = 2
            let subDeltaTime = deltaTime / Float(subSteps)

            for _ in 0..<subSteps {
                for i in 0..<16 {
                    var pos = balls[i].position
                    var vel = balls[i].velocity
                    var quat = balls[i].quaternion

                    if vel.x.isInfinite { continue }

                    pos += vel * subDeltaTime

                    let angVel3D = SIMD3<Float>(0.0, vel.x / ballRadius, -vel.y / ballRadius)
                    let angSpeed = length(angVel3D)
                    if angSpeed > 0.0 {
                        let axis = angVel3D / angSpeed
                        let deltaQuat = quaternionFromAxisAngle(axis, angSpeed * subDeltaTime)
                        quat = quaternionMultiply(deltaQuat, quat)
                        quat = normalize(quat)
                    }

                    if checkPocket(pos: pos) {
                        vel = SIMD2<Float>(Float.infinity, Float.infinity)
                        pos = SIMD2<Float>(0.0, 0.0)
                    } else {
                        if abs(pos.x) > tableWidth - ballRadius {
                            pos.x = (pos.x > 0 ? tableWidth - ballRadius : -tableWidth + ballRadius)
                            vel.x = -vel.x * restitution
                            let angVel = vel.y / ballRadius
                            let deltaQuat = quaternionFromAxisAngle(SIMD3<Float>(0.0, 0.0, 1.0), angVel * subDeltaTime)
                            quat = quaternionMultiply(deltaQuat, quat)
                            quat = normalize(quat)
                        }
                        if abs(pos.y) > tableLength - ballRadius {
                            pos.y = (pos.y > 0 ? tableLength - ballRadius : -tableLength + ballRadius)
                            vel.y = -vel.y * restitution
                            let angVel = -vel.x / ballRadius
                            let deltaQuat = quaternionFromAxisAngle(SIMD3<Float>(0.0, 1.0, 0.0), angVel * subDeltaTime)
                            quat = quaternionMultiply(deltaQuat, quat)
                            quat = normalize(quat)
                        }
                    }

                    vel *= friction
                    if length(vel) < minVelocity {
                        vel = SIMD2<Float>(0.0, 0.0)
                    }

                    balls[i].position = pos
                    balls[i].velocity = vel
                    balls[i].quaternion = quat
                }

                for i in 0..<15 {
                    for j in (i + 1)..<16 {
                        let pos1 = balls[i].position
                        let pos2 = balls[j].position
                        var vel1 = balls[i].velocity
                        var vel2 = balls[j].velocity
                        var quat1 = balls[i].quaternion
                        var quat2 = balls[j].quaternion

                        if vel1.x.isInfinite || vel2.x.isInfinite { continue }

                        let delta = pos2 - pos1
                        let dist = length(delta)

                        if dist < (2.0 * ballRadius) {
                            let normal = normalize(delta)
                            let relativeVel = vel1 - vel2
                            let impulse = dot(relativeVel, normal)

                            if impulse > 0.0 {
                                let impulseMag = -(1.0 + restitution) * impulse / (2.0 / ballMass)
                                vel1 += normal * impulseMag / ballMass
                                vel2 -= normal * impulseMag / ballMass

                                let tangent = SIMD2<Float>(-normal.y, normal.x)
                                let relVelTangent = dot(relativeVel, tangent)
                                let frictionImpulse = relVelTangent * 0.2

                                let contactPoint = normal * ballRadius
                                let r1 = -contactPoint
                                let r2 = contactPoint
                                let angVel1 = frictionImpulse / momentOfInertia
                                let angVel2 = -frictionImpulse / momentOfInertia

                                let axis1 = SIMD3<Float>(tangent.x, tangent.y, 0.0)
                                let axis2 = -axis1
                                let deltaQuat1 = quaternionFromAxisAngle(axis1, angVel1 * subDeltaTime)
                                let deltaQuat2 = quaternionFromAxisAngle(axis2, angVel2 * subDeltaTime)

                                quat1 = quaternionMultiply(deltaQuat1, quat1)
                                quat2 = quaternionMultiply(deltaQuat2, quat2)
                                quat1 = normalize(quat1)
                                quat2 = normalize(quat2)

                                let overlap = (2.0 * ballRadius) - dist
                                let correction = normal * (overlap * 0.5)
                                balls[i].position -= correction
                                balls[j].position += correction

                                balls[i].velocity = vel1
                                balls[j].velocity = vel2
                                balls[i].quaternion = quat1
                                balls[j].quaternion = quat2
                            }
                        }
                    }
                }

                for i in 0..<16 {
                    ballShaderData[i] = SIMD8<Float>(
                        balls[i].position.x, balls[i].position.y,
                        balls[i].velocity.x, balls[i].velocity.y,
                        balls[i].quaternion.x, balls[i].quaternion.y,
                        balls[i].quaternion.z, balls[i].quaternion.w
                    )
                }
            }
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            guard let device = device,
                  let pipelineState = pipelineState,
                  let commandQueue = commandQueue,
                  let drawable = view.currentDrawable,
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let renderPassDescriptor = view.currentRenderPassDescriptor,
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
                  let ballBuffer = ballBuffer,
                  let cueBuffer = cueBuffer,
                  let showCueBuffer = showCueBuffer
            else { return }

            let deltaTime = min(parent.time - lastFrameTime, 0.1)
            lastFrameTime = parent.time

            updatePhysics(deltaTime: deltaTime)

            var resolution = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))
            encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
            encoder.setFragmentBytes(&lastFrameTime, length: MemoryLayout<Float>.size, index: 1)
            encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 2)
            memcpy(cueBuffer.contents(), &cueOffset, MemoryLayout<Float>.stride)
            encoder.setFragmentBuffer(cueBuffer, offset: 0, index: 3)
            memcpy(showCueBuffer.contents(), &showCueValue, MemoryLayout<Int32>.stride)
            encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 4)

            encoder.setRenderPipelineState(pipelineState)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()

            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.delegate = context.coordinator
        view.device = MTLCreateSystemDefaultDevice()
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        view.preferredFramesPerSecond = 60
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}

// MARK: - ContentView
struct ContentView: View {
    @State private var time: Float = 0
    @State private var isTouching: Bool = false

    var body: some View {
        MetalView(time: $time, isTouching: $isTouching)
            .ignoresSafeArea()
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in isTouching = true }
                    .onEnded { _ in isTouching = false }
            )
            .onAppear {
                let timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { _ in
                    time += 1.0 / 60.0
                }
                RunLoop.current.add(timer, forMode: .common)
            }
    }
}

#Preview {
    ContentView()
}
