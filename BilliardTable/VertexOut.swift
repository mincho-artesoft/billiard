import SwiftUI
import MetalKit

// MARK: - Metal Shader
let metalShader = """
#include <metal_stdlib>
using namespace metal;

// ----------------------
// Data Structures
// ----------------------
struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Ball {
    float2 position;
    float2 velocity;
};

// ----------------------
// Utility Functions
// ----------------------
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
    return min(
        min(
            max(dxy + rt, dz),
            max(dxy, dz + rt)
        ),
        length(float2(dxy, dz) + rt) - rt
    );
}

float2 rot2D(float2 q, float a) {
    float2 cs = float2(cos(a), sin(a));
    return float2(q.x * cs.x - q.y * cs.y,
                  q.x * cs.y + q.y * cs.x);
}

float smoothMin(float a, float b, float r) {
    float h = clamp(0.5 + 0.5 * (b - a) / r, 0.0, 1.0);
    return mix(b, a, h) - r * h * (1.0 - h);
}

float smoothMax(float a, float b, float r) {
    return -smoothMin(-a, -b, r);
}

// ----------------------
// Ball Ray Intersection
// ----------------------
void ballHit(
    float3 ro,
    float3 rd,
    thread float &dist,
    thread float3 &normal,
    thread int &id,
    constant Ball* balls [[buffer(2)]]
)
{
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

// ----------------------
// Scene SDF + Shading
// ----------------------
float3 showScene(
    float3 ro,
    float3 rd,
    float time,
    float cueOffset,
    constant Ball* balls [[buffer(2)]],
    int cueVisible
)
{
    const float hbLen = 8.0;
    const float bWid  = 0.4;
    const float2 hIn  = float2(hbLen, hbLen * 1.75) - bWid;
    const float PI    = 3.14159;

    float3 col = float3(0.05, 0.05, 0.1);
    float t = 0.0;
    const float maxDist = 50.0;

    // Ball ray intersection
    float dstBall;
    float3 ballNormal;
    int ballId;
    ballHit(ro, rd, dstBall, ballNormal, ballId, balls);

    float dstTable = maxDist;
    float dstCue   = maxDist;
    float3 cueHitPos;

    // Raymarch for the table and the cue
    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * t;

        // Table geometry
        float dSurface = prBoxDf(p, float3(hIn.x, 0.4, hIn.y));
        float3 pb = p; 
        pb.y -= -0.6;
        float dBorder = prRoundBoxDf(pb, float3(hIn.x + 0.6, 0.5, hIn.y + 0.6), 0.2);
        float dTable = max(dBorder, -dSurface);

        // Pocket shaping
        float2 hInPocket = hIn - bWid + 0.03;
        float3 q = p;
        q.x = abs(q.x) - hInPocket.x;
        q.z = fmod(q.z + 0.5 * hInPocket.y, hInPocket.y) - 0.5 * hInPocket.y;
        float pocketDist = length(q.xz);
        dTable = smoothMax(dTable, 0.53 - pocketDist, 0.01);

        // Cue geometry
        float3 pc = p - float3(balls[0].position.x, 0.05, balls[0].position.y);
        pc.y -= 0.05;
        float aCue = sin(time * 0.5) * 0.1;
        pc.xz = -rot2D(pc.xz, aCue);
        float cueLength = 2.5;
        float ballRadius = 0.47;
        float tipOffset = cueLength;
        float maxCueOffset = -ballRadius;

        // Move the cue in/out
        pc.z -= (maxCueOffset - cueOffset - tipOffset);

        float dCueStick = prRoundCylDf(
            pc,
            0.1 - (0.015 / 2.5) * (pc.z + tipOffset),
            0.05,
            cueLength
        );

        // If the cue is hidden, push it far away
        if (cueVisible == 0) {
            dCueStick = 99999.0;
        }

        float d = min(dTable, dCueStick);

        // If we collide with table/cue or pass the ball distance, break
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
        // -- Hit the ball
        if (dstBall <= min(dstTable, dstCue)) {
            float3 p = ro + rd * dstBall;
            float3 n = ballNormal;
            int id = ballId;

            // Ball coloring
            if (id == 0) {
                // Cue ball
                col = float3(1.0);
            } else {
                // Numbered balls
                float c = float(id - 1);
                float3 baseColor;
                bool isStriped = (id >= 9);

                // 8-ball is black
                if (id == 8) {
                    baseColor = float3(0.0);
                } else {
                    baseColor = hsvToRgb(float3(
                        fmod(c / 7.0, 1.0),
                        1.0,
                        1.0
                    ));
                }

                // Spherical UV
                float2 uv = float2(
                    atan2(n.x, n.z) / (2.0 * PI) + 0.5,
                    acos(n.y) / PI
                );
                // Stripe check
                if (isStriped && id != 8) {
                    float stripeFactor = sin(uv.y * PI * 10.0);
                    col = mix(float3(1.0), baseColor, step(0.0, stripeFactor));
                } else {
                    col = baseColor;
                }

                // "Number" circle
                float2 circleCenter = float2(0.5, 0.5);
                float circleRadius = 0.2;
                float distToCenter = length(uv - circleCenter);
                if (distToCenter < circleRadius && id != 0) {
                    // Outer ring white
                    col = float3(1.0);
                    // Inner circle black
                    if (distToCenter < circleRadius * 0.7) {
                        col = float3(0.0);
                    }
                }
            }
            // Lighting
            col *= 0.2 + 0.8 * max(n.y, 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(
                max(dot(r, normalize(lightPos - p)), 0.0),
                16.0
            );
            col += float3(0.2) * spec;
        }
        // -- Hit the cue
        else if (dstCue < dstTable) {
            float3 p = ro + rd * dstCue;
            float3 eps = float3(0.001, 0.0, 0.0);
            float3 n = normalize(float3(
                prRoundCylDf(cueHitPos + eps.xyy, 0.1, 0.05, 2.5)
              - prRoundCylDf(cueHitPos - eps.xyy, 0.1, 0.05, 2.5),
                prRoundCylDf(cueHitPos + eps.yxy, 0.1, 0.05, 2.5)
              - prRoundCylDf(cueHitPos - eps.yxy, 0.1, 0.05, 2.5),
                prRoundCylDf(cueHitPos + eps.yyx, 0.1, 0.05, 2.5)
              - prRoundCylDf(cueHitPos - eps.yyx, 0.1, 0.05, 2.5)
            ));
            col = (cueHitPos.z < 2.2)
                ? float3(0.5, 0.3, 0.0)  // "wood" handle
                : float3(0.7, 0.7, 0.3); // lighter tip

            float diff = max(dot(n, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(
                max(dot(r, normalize(lightPos - p)), 0.0),
                16.0
            );
            col *= 0.3 + 0.7 * diff;
            col += float3(0.2) * spec;
        }
        // -- Hit the table
        else {
            float3 p = ro + rd * dstTable;
            float3 eps = float3(0.001, 0.0, 0.0);
            float3 n = normalize(float3(
                prBoxDf(p + eps.xyy, float3(hIn.x, 0.4, hIn.y))
              - prBoxDf(p - eps.xyy, float3(hIn.x, 0.4, hIn.y)),
                prBoxDf(p + eps.yxy, float3(hIn.x, 0.4, hIn.y))
              - prBoxDf(p - eps.yxy, float3(hIn.x, 0.4, hIn.y)),
                prBoxDf(p + eps.yyx, float3(hIn.x, 0.4, hIn.y))
              - prBoxDf(p - eps.yyx, float3(hIn.x, 0.4, hIn.y))
            ));

            float2 pocketCheck = float2(
                abs(p.x) - (hIn.x - bWid + 0.03),
                fmod(p.z + 0.5 * (hIn.y - bWid + 0.03),
                     (hIn.y - bWid + 0.03)) - 0.5 * (hIn.y - bWid + 0.03)
            );
            float pocketDist = length(pocketCheck);

            if (pocketDist < 0.53) {
                col = float3(0.0); // inside a pocket
            } else if (max(abs(p.x) - hIn.x, abs(p.z) - hIn.y) < 0.3) {
                col = float3(0.1, 0.5, 0.3); // felt
            } else {
                col = float3(0.3, 0.1, 0.0); // wooden sides
            }

            float diff = max(dot(n, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(
                max(dot(r, normalize(lightPos - p)), 0.0),
                16.0
            );
            col *= 0.3 + 0.7 * diff;
            col += float3(0.2) * spec;
        }
    }

    return clamp(col, 0.0, 1.0);
}

// ----------------------
// Vertex Shader
// ----------------------
vertex VertexOut vertexShader(uint vertexID [[vertex_id]])
{
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

// ----------------------
// Fragment Shader
// ----------------------
fragment float4 fragmentShader(
    VertexOut in [[stage_in]],
    constant float2 &resolution  [[buffer(0)]],
    constant float  &time        [[buffer(1)]],
    constant Ball*  balls        [[buffer(2)]],
    constant float  &cueOffset   [[buffer(3)]],
    constant int    &cueVisible  [[buffer(4)]]
)
{
    float2 uv = 2.0 * in.uv - 1.0;
    uv.x *= resolution.x / resolution.y;

    float3 target = float3(0.0, 0.0, 0.0);
    float3 ro = float3(0.0, 10.0, 20.0);

    float angle = time * 0.1;
    ro = float3(sin(angle) * 20.0, 10.0, cos(angle) * 20.0);

    float3 vd = normalize(target - ro);
    float3 right = normalize(cross(float3(0.0, 1.0, 0.0), vd));
    float3 up    = normalize(cross(vd, right));

    const float fov = 0.8;
    float3 rd = normalize(
        vd + right * uv.x * fov + up * uv.y * fov
    );

    float3 col = showScene(ro, rd, time, cueOffset, balls, cueVisible);
    return float4(col, 1.0);
}
"""

// MARK: - Metal View
struct MetalView: UIViewRepresentable {
    @Binding var time: Float
    @Binding var isTouching: Bool
    
    class Coordinator: NSObject, MTKViewDelegate {
        private let parent: MetalView
        private let device: MTLDevice?
        private let commandQueue: MTLCommandQueue?
        private var pipelineState: MTLRenderPipelineState?
        
        // Time tracking
        private var lastFrameTime: Float
        
        // Ball data
        private var balls: [SIMD4<Float>]
        private var ballBuffer: MTLBuffer?
        
        // Cue data
        private var cueOffset: Float = 0.0
        private var cueBuffer: MTLBuffer?
        
        // Cue visibility data
        private var showCueValue: Int32 = 1 // 1 = show, 0 = hide
        private var showCueBuffer: MTLBuffer?
        
        // Table constants
        private let ballRadius: Float = 0.47
        private let tableWidth: Float = 7.6
        private let tableLength: Float = 13.3
        private let pocketRadius: Float = 0.53
        
        // Cue speeds
        private let cuePullSpeed: Float = 1.0   // Pull-back
        private let cueStrikeSpeed: Float = 10.0 // Strike
        private let maxCueOffset: Float = 2.5   // Max pull-back distance
        
        // Collision/physics
        private var hitTriggered: Bool = false
        private var shooting: Bool = false
        private var powerAtRelease: Float = 0.0
        
        // Tuned friction/restitution to ensure balls actually stop
        private let friction: Float = 0.92
        private let restitution: Float = 0.75
        private let minVelocity: Float = 0.05
        
        init(parent: MetalView) {
            self.parent = parent
            self.device = MTLCreateSystemDefaultDevice()
            self.commandQueue = device?.makeCommandQueue()
            self.lastFrameTime = parent.time
            // 16 balls:
            self.balls = [
                // x, y, vx, vy
                SIMD4<Float>( 0.0,  5.0,  0.0,  0.0),  // Cue ball
                SIMD4<Float>(-0.5, -2.0,  0.0,  0.0),
                SIMD4<Float>( 0.5, -2.0,  0.0,  0.0),
                SIMD4<Float>(-1.0, -1.5, 0.0,  0.0),
                SIMD4<Float>( 0.0, -1.5, 0.0,  0.0),
                SIMD4<Float>( 1.0, -1.5, 0.0,  0.0),
                SIMD4<Float>(-1.5, -1.0, 0.0,  0.0),
                SIMD4<Float>(-0.5, -1.0, 0.0,  0.0),
                SIMD4<Float>( 0.5, -1.0, 0.0,  0.0),
                SIMD4<Float>( 1.5, -1.0, 0.0,  0.0),
                SIMD4<Float>(-2.0, -0.5, 0.0,  0.0),
                SIMD4<Float>(-1.0, -0.5, 0.0,  0.0),
                SIMD4<Float>( 0.0, -0.5, 0.0,  0.0),
                SIMD4<Float>( 1.0, -0.5, 0.0,  0.0),
                SIMD4<Float>( 2.0, -0.5, 0.0,  0.0),
                SIMD4<Float>( 0.0,  0.0,  0.0,  0.0)
            ]
            
            super.init()
            
            // Create buffers, compile pipeline
            if let device = device {
                // Balls
                ballBuffer = device.makeBuffer(
                    bytes: &balls,
                    length: MemoryLayout<SIMD4<Float>>.stride * 16,
                    options: .storageModeShared
                )
                
                // Cue offset
                cueBuffer = device.makeBuffer(
                    length: MemoryLayout<Float>.stride,
                    options: .storageModeShared
                )
                
                // Cue visibility
                showCueBuffer = device.makeBuffer(
                    length: MemoryLayout<Int32>.stride,
                    options: .storageModeShared
                )
                
                do {
                    let library = try device.makeLibrary(source: metalShader, options: nil)
                    
                    guard let vertexFunction = library.makeFunction(name: "vertexShader") else {
                        print("Vertex shader function not found.")
                        return
                    }
                    guard let fragmentFunction = library.makeFunction(name: "fragmentShader") else {
                        print("Fragment shader function not found.")
                        return
                    }
                    
                    let descriptor = MTLRenderPipelineDescriptor()
                    descriptor.vertexFunction = vertexFunction
                    descriptor.fragmentFunction = fragmentFunction
                    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
                    
                    pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
                } catch {
                    print("Failed to create pipeline state: \(error)")
                    pipelineState = nil
                }
            }
        }
        
        // Check if ball is pocketed
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
        
        // Physics update each frame
        func updatePhysics(deltaTime: Float) {
            guard let ballBuffer = ballBuffer else { return }
            let ballData = ballBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: 16)
            
            // Cue stick control (pull back / shoot)
            if parent.isTouching && !shooting {
                cueOffset += cuePullSpeed * deltaTime
                if cueOffset > maxCueOffset {
                    cueOffset = maxCueOffset
                }
            }
            else if !parent.isTouching && cueOffset > 0.0 {
                // Released => strike
                if !shooting {
                    powerAtRelease = cueOffset / maxCueOffset
                    shooting = true
                }
                // Move forward quickly
                cueOffset -= cueStrikeSpeed * deltaTime
                if cueOffset <= 0.0 {
                    cueOffset = 0.0
                    // Apply velocity to the cue ball
                    let baseVelocity: Float = -10.0
                    ballData[0].z = 0.0
                    ballData[0].w = baseVelocity * (1.0 + powerAtRelease) // up to ~ -20
                    hitTriggered = true
                    shooting = false
                    powerAtRelease = 0.0
                    
                    // Hide cue after striking
                    showCueValue = 0
                }
            }
            
            // If we've struck the ball, bring back the cue once all balls stop
            if !parent.isTouching && hitTriggered {
                var allStopped = true
                for i in 0..<16 {
                    let vel = SIMD2<Float>(ballData[i].z, ballData[i].w)
                    // If pocketed => skip
                    if vel.x.isInfinite { continue }
                    
                    if length(vel) > minVelocity {
                        allStopped = false
                        break
                    }
                }
                if allStopped {
                    // All are at rest or pocketed
                    hitTriggered = false
                    showCueValue = 1 // Re-show the cue
                }
            }
            
            // Update ball positions and handle collisions
            for i in 0..<16 {
                var pos = SIMD2<Float>(ballData[i].x, ballData[i].y)
                var vel = SIMD2<Float>(ballData[i].z, ballData[i].w)
                
                // Pocketed => skip
                if vel.x.isInfinite { continue }
                
                // Move
                pos += vel * deltaTime
                
                // Check pockets
                if checkPocket(pos: pos) {
                    // Mark as pocketed
                    vel = SIMD2<Float>(Float.infinity, Float.infinity)
                    pos = SIMD2<Float>(0.0, 0.0)
                } else {
                    // Cushion collisions
                    if abs(pos.x) > tableWidth - ballRadius {
                        pos.x = (pos.x > 0
                                 ? tableWidth - ballRadius
                                 : -tableWidth + ballRadius)
                        vel.x = -vel.x * restitution
                    }
                    if abs(pos.y) > tableLength - ballRadius {
                        pos.y = (pos.y > 0
                                 ? tableLength - ballRadius
                                 : -tableLength + ballRadius)
                        vel.y = -vel.y * restitution
                    }
                }
                
                // Apply friction
                vel *= friction
                // Snap small velocity to zero
                if length(vel) < 0.01 {
                    vel = SIMD2<Float>(0.0, 0.0)
                }
                
                // Write results
                ballData[i].x = pos.x
                ballData[i].y = pos.y
                ballData[i].z = vel.x
                ballData[i].w = vel.y
            }
            
            // Ball-ball collisions
            for i in 0..<15 {
                for j in (i + 1)..<16 {
                    let pos1 = SIMD2<Float>(ballData[i].x, ballData[i].y)
                    let pos2 = SIMD2<Float>(ballData[j].x, ballData[j].y)
                    let vel1 = SIMD2<Float>(ballData[i].z, ballData[i].w)
                    let vel2 = SIMD2<Float>(ballData[j].z, ballData[j].w)
                    
                    if vel1.x.isInfinite || vel2.x.isInfinite { continue }
                    
                    let delta = pos2 - pos1
                    let dist = length(delta)
                    
                    // Overlap => collide
                    if dist < (2.0 * ballRadius) {
                        let normal = normalize(delta)
                        let relativeVel = vel2 - vel1
                        let impulse = dot(relativeVel, normal)
                        if impulse < 0.0 {
                            // Collide with restitution
                            let combinedRest = 0.75
                            let impulseMag = -(Float(1.0) + Float(combinedRest)) * impulse / Float(2.0)
                            ballData[i].z += impulseMag * normal.x
                            ballData[i].w += impulseMag * normal.y
                            ballData[j].z -= impulseMag * normal.x
                            ballData[j].w -= impulseMag * normal.y
                            
                            // Push them out of overlap
                            let overlap = (2.0 * ballRadius) - dist
                            ballData[i].x -= normal.x * (overlap * 0.5)
                            ballData[i].y -= normal.y * (overlap * 0.5)
                            ballData[j].x += normal.x * (overlap * 0.5)
                            ballData[j].y += normal.y * (overlap * 0.5)
                        }
                    }
                }
            }
        }
        
        // MTKView delegate
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
            else {
                return
            }
            
            let deltaTime = min(parent.time - lastFrameTime, 0.1)
            lastFrameTime = parent.time
            
            updatePhysics(deltaTime: deltaTime)
            
            // --- Update uniform/buffer data ---
            // 1) Resolution
            var resolution = SIMD2<Float>(
                Float(view.drawableSize.width),
                Float(view.drawableSize.height)
            )
            encoder.setFragmentBytes(&resolution,
                                     length: MemoryLayout<SIMD2<Float>>.size,
                                     index: 0)
            
            // 2) Time
            encoder.setFragmentBytes(&lastFrameTime,
                                     length: MemoryLayout<Float>.size,
                                     index: 1)
            
            // 3) Balls
            encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 2)
            
            // 4) Cue Offset
            memcpy(cueBuffer.contents(), &cueOffset, MemoryLayout<Float>.stride)
            encoder.setFragmentBuffer(cueBuffer, offset: 0, index: 3)
            
            // 5) Cue Visibility
            memcpy(showCueBuffer.contents(), &showCueValue, MemoryLayout<Int32>.stride)
            encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 4)
            
            // --- Draw ---
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
        view.clearColor = MTLClearColor(red: 0.05,
                                        green: 0.05,
                                        blue: 0.1,
                                        alpha: 1.0)
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
                    .onChanged { _ in
                        isTouching = true
                    }
                    .onEnded { _ in
                        isTouching = false
                    }
            )
            .onAppear {
                let timer = Timer.scheduledTimer(withTimeInterval: 1 / 60, repeats: true) { _ in
                    time += 1 / 60
                }
                RunLoop.current.add(timer, forMode: .common)
            }
    }
}

// MARK: - SwiftUI Preview
#Preview {
    ContentView()
}
