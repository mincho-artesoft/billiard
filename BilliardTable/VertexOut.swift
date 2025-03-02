//
//  VertexOut.swift
//  BilliardTable
//
//  Created by Mincho Milev on 3/2/25.
//


import SwiftUI
import MetalKit

// MARK: - Metal Shader
let metalShader = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// --------------------------------------------------
// Utility Functions
// --------------------------------------------------

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
    float dxy = length(p.xy) - (r - (rt / 2.5) * p.z);  // Tapering radius
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

// --------------------------------------------------
// Ball Handling
// --------------------------------------------------

void ballHit(float3 ro, float3 rd, thread float& dist, thread float3& normal, thread int& id) {
    const int nBall = 16;
    const float rad = 0.47;
    
    float2 ballPos[nBall] = {
        float2(0.0, 5.0),    // Cue ball
        float2(-0.5, -2.0),  // Rack start
        float2(0.5, -2.0),
        float2(-1.0, -1.5),
        float2(0.0, -1.5),
        float2(1.0, -1.5),
        float2(-1.5, -1.0),
        float2(-0.5, -1.0),
        float2(0.5, -1.0),
        float2(1.5, -1.0),
        float2(-2.0, -0.5),
        float2(-1.0, -0.5),
        float2(0.0, -0.5),
        float2(1.0, -0.5),
        float2(2.0, -0.5),
        float2(0.0, 0.0)     // Apex of rack
    };
    
    dist = 50.0;
    normal = float3(0.0);
    id = -1;
    
    for (int n = 0; n < nBall; n++) {
        float3 u = ro - float3(ballPos[n].x, 0.05, ballPos[n].y);
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

// --------------------------------------------------
// Scene SDF + Shading
// --------------------------------------------------
float3 showScene(float3 ro, float3 rd, float time)
{
    const float hbLen = 8.0;
    const float bWid = 0.4;
    const float2 hIn = float2(hbLen, hbLen * 1.75) - bWid; // x (short) = 7.6, z (long) = 13.3

    float3 col = float3(0.05, 0.05, 0.1);
    float t = 0.0;
    const float maxDist = 50.0;

    // Check balls
    float dstBall;
    float3 ballNormal;
    int ballId;
    ballHit(ro, rd, dstBall, ballNormal, ballId);
    
    // Table and cue ray marching
    float dstTable = maxDist;
    float dstCue = maxDist;
    float3 cueHitPos;
    
    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * t;
        
        // Table surface
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
        
        // Cue stick
        float3 pc = p;
        pc.yz -= float2(0.0, -0.6 * (hIn.y + bWid));  // Position above table
        float aCue = sin(time * 0.5) * 0.3;  // Simple animation
        float dCue = sin(time * 0.2) * 0.5;  // Simple oscillation
        pc.xz = rot2D(pc.xz, 0.5 * 3.14159 - aCue);  // Rotate around Z
        pc.z -= -3.05 - dCue;  // Shift along Z
        float dCueStick = prRoundCylDf(pc, 0.1, 0.05, 2.5);
        
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
            // Ball hit
            float3 p = ro + rd * dstBall;
            float3 n = ballNormal;
            int id = ballId;
            
            if (id == 0) {
                col = float3(1.0);  // Cue ball
            } else {
                float c = float(id - 1);
                col = hsvToRgb(float3(
                    fmod(c / 15.0, 1.0),
                    1.0 - 0.3 * fmod(c, 3.0),
                    1.0 - 0.2 * fmod(c, 2.0)
                ));
                col *= (n.y > 0.0 ? 1.0 : 0.4);
            }
            
            col *= 0.2 + 0.8 * max(n.y, 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col += float3(0.2) * spec;
        } else if (dstCue < dstTable) {
            // Cue stick hit
            float3 p = ro + rd * dstCue;
            float3 eps = float3(0.001, 0.0, 0.0);
            float3 n = normalize(float3(
                prRoundCylDf(cueHitPos + eps.xyy, 0.1, 0.05, 2.5) - prRoundCylDf(cueHitPos - eps.xyy, 0.1, 0.05, 2.5),
                prRoundCylDf(cueHitPos + eps.yxy, 0.1, 0.05, 2.5) - prRoundCylDf(cueHitPos - eps.yxy, 0.1, 0.05, 2.5),
                prRoundCylDf(cueHitPos + eps.yyx, 0.1, 0.05, 2.5) - prRoundCylDf(cueHitPos - eps.yyx, 0.1, 0.05, 2.5)
            ));
            
            col = (cueHitPos.z < 2.2) ? float3(0.5, 0.3, 0.0) : float3(0.7, 0.7, 0.3);  // Shaft vs tip
            float diff = max(dot(n, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col *= 0.3 + 0.7 * diff;
            col += float3(0.2) * spec;
        } else {
            // Table hit
            float3 p = ro + rd * dstTable;
            float3 eps = float3(0.001, 0.0, 0.0);
            float3 n = normalize(float3(
                prBoxDf(p + eps.xyy, float3(hIn.x, 0.4, hIn.y)) - prBoxDf(p - eps.xyy, float3(hIn.x, 0.4, hIn.y)),
                prBoxDf(p + eps.yxy, float3(hIn.x, 0.4, hIn.y)) - prBoxDf(p - eps.yxy, float3(hIn.x, 0.4, hIn.y)),
                prBoxDf(p + eps.yyx, float3(hIn.x, 0.4, hIn.y)) - prBoxDf(p - eps.yyx, float3(hIn.x, 0.4, hIn.y))
            ));
            
            float2 pocketCheck = float2(abs(p.x) - (hIn.x - bWid + 0.03), fmod(p.z + 0.5 * (hIn.y - bWid + 0.03), hIn.y - bWid + 0.03) - 0.5 * (hIn.y - bWid + 0.03));
            float pocketDist = length(pocketCheck);
            
            if (pocketDist < 0.53) {
                col = float3(0.0, 0.0, 0.0);
            }
            else if (max(abs(p.x) - hIn.x, abs(p.z) - hIn.y) < 0.3) {
                col = float3(0.1, 0.5, 0.3);
            }
            else {
                col = float3(0.3, 0.1, 0.0);
            }
            
            float diff = max(dot(n, normalize(lightPos - p)), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, normalize(lightPos - p)), 0.0), 16.0);
            col *= 0.3 + 0.7 * diff;
            col += float3(0.2) * spec;
        }
    }
    
    col += hsvToRgb(float3(time * 0.1, 0.3, 0.05)) * 0.02;
    return clamp(col, 0.0, 1.0);
}

// --------------------------------------------------
// Vertex Shader
// --------------------------------------------------
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

// --------------------------------------------------
// Fragment Shader
// --------------------------------------------------
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant float2 &resolution [[buffer(0)]],
                               constant float &time       [[buffer(1)]])
{
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
    
    float3 col = showScene(ro, rd, time);
    return float4(col, 1.0);
}
"""

// MARK: - Metal View
struct MetalView: UIViewRepresentable {
    @Binding var time: Float
    
    class Coordinator: NSObject, MTKViewDelegate {
        private let parent: MetalView
        private let device: MTLDevice?
        private let commandQueue: MTLCommandQueue?
        private let pipelineState: MTLRenderPipelineState?
        private var lastFrameTime: Float
        
        init(parent: MetalView) {
            self.parent = parent
            self.device = MTLCreateSystemDefaultDevice()
            self.commandQueue = device?.makeCommandQueue()
            self.lastFrameTime = parent.time
            
            var pipeline: MTLRenderPipelineState? = nil
            if let device = device {
                do {
                    let library = try device.makeLibrary(source: metalShader, options: nil)
                    let vertexFunction = library.makeFunction(name: "vertexShader")
                    let fragmentFunction = library.makeFunction(name: "fragmentShader")
                    
                    let descriptor = MTLRenderPipelineDescriptor()
                    descriptor.vertexFunction = vertexFunction
                    descriptor.fragmentFunction = fragmentFunction
                    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
                    
                    pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
                } catch {
                    print("Failed to create pipeline state: \\(error)")
                }
            }
            self.pipelineState = pipeline
            super.init()
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func draw(in view: MTKView) {
            guard
                let device = device,
                let pipelineState = pipelineState,
                let commandQueue = commandQueue,
                let drawable = view.currentDrawable,
                let commandBuffer = commandQueue.makeCommandBuffer(),
                let renderPassDescriptor = view.currentRenderPassDescriptor,
                let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
            else {
                return
            }
            
            lastFrameTime = parent.time
            encoder.setRenderPipelineState(pipelineState)
            
            var resolution = float2(Float(view.drawableSize.width),
                                    Float(view.drawableSize.height))
            encoder.setFragmentBytes(&resolution, length: MemoryLayout<float2>.size, index: 0)
            encoder.setFragmentBytes(&lastFrameTime, length: MemoryLayout<Float>.size, index: 1)
            
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
    
    var body: some View {
        MetalView(time: $time)
            .ignoresSafeArea()
            .onAppear {
                let timer = Timer.scheduledTimer(withTimeInterval: 1/60, repeats: true) { _ in
                    time += 1/60
                }
                RunLoop.current.add(timer, forMode: .common)
            }
    }
}

#Preview {
    ContentView()
}