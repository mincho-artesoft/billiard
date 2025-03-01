import SwiftUI
import MetalKit
import simd

//==================================================
// MARK: - Metal Shaders (Embedded as Swift String)
//==================================================
fileprivate let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

// Vertex structure with position, normal, uv.
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

// Uniforms from Swift
struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float3   lightPosition;
};

// Vertex shader output
struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float3 worldPos;
    float2 uv;
    float  materialId;
};

vertex VertexOut vertexShader(VertexIn inVertex [[stage_in]],
                              constant Uniforms &uniforms [[buffer(1)]],
                              constant float &matID [[buffer(2)]])
{
    VertexOut outVertex;
    
    float4 worldPosition = uniforms.modelMatrix * float4(inVertex.position, 1.0);
    outVertex.position   = uniforms.projectionMatrix * (uniforms.viewMatrix * worldPosition);
    
    float3 worldNormal   = (uniforms.modelMatrix * float4(inVertex.normal, 0.0)).xyz;
    outVertex.normal     = normalize(worldNormal);
    outVertex.worldPos   = worldPosition.xyz;
    outVertex.uv         = inVertex.uv;
    outVertex.materialId = matID;
    
    return outVertex;
}

fragment float4 fragmentShader(VertexOut inVertex [[stage_in]],
                               // We'll have 2 textures: felt & wood
                               texture2d<float> feltTex  [[texture(0)]],
                               texture2d<float> woodTex  [[texture(1)]],
                               sampler                samp [[sampler(0)]])
{
    // Simple Lambertian lighting
    float3 normal = normalize(inVertex.normal);
    float3 lightPos = float3(0.0, 2.0, 2.0);
    float3 lightDir = normalize(lightPos - inVertex.worldPos);
    float NdotL     = max(dot(normal, lightDir), 0.0);
    
    // We'll pick which texture to sample from materialId:
    float2 uv = inVertex.uv;
    float3 textureColor;
    if (inVertex.materialId < 0.5) {
        // Felt
        float4 c = feltTex.sample(samp, uv);
        textureColor = c.rgb;
    } else if (inVertex.materialId < 1.5) {
        // Wood
        float4 c = woodTex.sample(samp, uv);
        textureColor = c.rgb;
    } else {
        // Some non-textured fallback (just color).
        // Let’s pick a bright color, e.g. red:
        textureColor = float3(1.0, 0.0, 0.0);
    }
    
    // Basic ambient + diffuse
    float3 ambient = 0.2 * textureColor;
    float3 diffuse = 0.8 * textureColor * NdotL;
    float3 final   = ambient + diffuse;
    
    return float4(final, 1.0);
}
"""

//==================================================
// MARK: - Data Structures
//==================================================

/// Our vertex format (position, normal, uv).
struct Vertex {
    var position: SIMD3<Float>
    var normal:   SIMD3<Float>
    var uv:       SIMD2<Float>
}

/// Holds uniforms to pass to the shaders.
struct Uniforms {
    var modelMatrix:      matrix_float4x4
    var viewMatrix:       matrix_float4x4
    var projectionMatrix: matrix_float4x4
    var lightPosition:    SIMD3<Float>
}

/// A simple orbit camera for 3D rotation & zoom.
struct Camera {
    var rotation:  Float = 0    // horizontal angle
    var pitch:     Float = 0    // vertical angle
    var distance:  Float = 3
    var center:    SIMD3<Float> = [0, 0, 0]
    
    var aspectRatio: Float = 1
    var fov:         Float = 60 * (.pi / 180)

    mutating func updateCameraDrag(deltaX: Float, deltaY: Float) {
        rotation += deltaX * 0.01
        pitch    -= deltaY * 0.01
        // clamp pitch so we don't flip under the table
        let maxPitch: Float = .pi * 0.49
        pitch = max(-maxPitch, min(maxPitch, pitch))
    }
    
    mutating func updateCameraZoom(pinchScale: Float) {
        distance = max(0.5, distance - pinchScale * 0.01)
    }

    var viewMatrix: matrix_float4x4 {
        let x = distance * cos(pitch) * sin(rotation)
        let y = distance * sin(pitch) + 1.5
        let z = distance * cos(pitch) * cos(rotation)
        let eye = SIMD3<Float>(x, y, z)
        return lookAt(eye: eye, center: center, up: SIMD3<Float>(0,1,0))
    }

    var projectionMatrix: matrix_float4x4 {
        return perspectiveFov(fovy: fov, aspect: aspectRatio, nearZ: 0.01, farZ: 100)
    }
    
    // MARK: - Helpers
    private func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> matrix_float4x4 {
        let f = simd_normalize(center - eye)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)

        var mat = matrix_identity_float4x4
        mat.columns.0 = SIMD4<Float>( s.x,  u.x, -f.x, 0)
        mat.columns.1 = SIMD4<Float>( s.y,  u.y, -f.y, 0)
        mat.columns.2 = SIMD4<Float>( s.z,  u.z, -f.z, 0)
        mat.columns.3 = SIMD4<Float>(-simd_dot(s, eye),
                                     -simd_dot(u, eye),
                                      simd_dot(f, eye),
                                      1)
        return mat
    }

    private func perspectiveFov(fovy: Float, aspect: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
        let yScale = 1 / tan(fovy * 0.5)
        let xScale = yScale / aspect
        let zRange = farZ - nearZ
        let zScale = -(farZ + nearZ) / zRange
        let wzScale = -2 * farZ * nearZ / zRange

        var mat = matrix_identity_float4x4
        mat.columns.0.x = xScale
        mat.columns.1.y = yScale
        mat.columns.2.z = zScale
        mat.columns.2.w = -1
        mat.columns.3.z = wzScale
        mat.columns.3.w = 0
        return mat
    }
}

//==================================================
// MARK: - BilliardTable
//==================================================
struct BilliardTable {
    var vertices: [Vertex] = []
    var indices:  [UInt16] = []
    var modelMatrix = matrix_identity_float4x4
    
    init() {
        // We'll build:
        // 1) The main playing surface (felt).
        // 2) The rails & pockets (wood).
        let (feltVerts, feltInds)    = buildFeltSurface()
        let (railVerts, railInds)    = buildRails(startIndex: UInt16(feltVerts.count))
        let (pocketVerts, pocketInds) = buildPockets(startIndex: UInt16(feltVerts.count + railVerts.count))
        
        vertices = feltVerts + railVerts + pocketVerts
        indices  = feltInds  + railInds  + pocketInds
    }

    private func buildFeltSurface() -> ([Vertex], [UInt16]) {
        // Table top
        let p1: SIMD3<Float> = [-1, 0, -2]
        let p2: SIMD3<Float> = [ 1, 0, -2]
        let p3: SIMD3<Float> = [ 1, 0,  2]
        let p4: SIMD3<Float> = [-1, 0,  2]
        
        let n: SIMD3<Float>  = [0, 1, 0]
        let v1: SIMD2<Float> = [0,0]
        let v2: SIMD2<Float> = [1,0]
        let v3: SIMD2<Float> = [1,1]
        let v4: SIMD2<Float> = [0,1]
        
        let verts = [
            Vertex(position: p1, normal: n, uv: v1),
            Vertex(position: p2, normal: n, uv: v2),
            Vertex(position: p3, normal: n, uv: v3),
            Vertex(position: p4, normal: n, uv: v4),
        ]
        let inds: [UInt16] = [0,1,2, 0,2,3]
        return (verts, inds)
    }

    private func buildRails(startIndex: UInt16) -> ([Vertex], [UInt16]) {
        var railVertices: [Vertex] = []
        var railIndices:  [UInt16] = []
        var current = startIndex
        
        func buildRailBox(minX: Float, maxX: Float,
                          minZ: Float, maxZ: Float,
                          yBottom: Float, yTop: Float) -> ([Vertex], [UInt16]) {
            let corners = [
                SIMD3<Float>(minX, yBottom, minZ),
                SIMD3<Float>(maxX, yBottom, minZ),
                SIMD3<Float>(maxX, yBottom, maxZ),
                SIMD3<Float>(minX, yBottom, maxZ),
                SIMD3<Float>(minX, yTop,    minZ),
                SIMD3<Float>(maxX, yTop,    minZ),
                SIMD3<Float>(maxX, yTop,    maxZ),
                SIMD3<Float>(minX, yTop,    maxZ),
            ]
            
            // Simplify normals to up
            let n  = SIMD3<Float>(0,1,0)
            let uv = SIMD2<Float>(0,0)
            
            let vs = corners.map { Vertex(position: $0, normal: n, uv: uv) }
            let iset: [UInt16] = [
                0,1,2, 0,2,3,
                4,5,6, 4,6,7,
                0,1,5, 0,5,4,
                2,3,7, 2,7,6,
                0,3,7, 0,7,4,
                1,2,6, 1,6,5
            ]
            return (vs, iset)
        }
        
        let left = buildRailBox(minX: -1.1, maxX: -1.0, minZ: -2.0, maxZ: 2.0, yBottom: 0.0, yTop: 0.1)
        railVertices.append(contentsOf: left.0)
        railIndices.append(contentsOf: left.1.map { $0 + current })
        current += UInt16(left.0.count)

        let right = buildRailBox(minX: 1.0, maxX: 1.1, minZ: -2.0, maxZ: 2.0, yBottom: 0.0, yTop: 0.1)
        railVertices.append(contentsOf: right.0)
        railIndices.append(contentsOf: right.1.map { $0 + current })
        current += UInt16(right.0.count)

        let top = buildRailBox(minX: -1.0, maxX: 1.0, minZ: -2.1, maxZ: -2.0, yBottom: 0.0, yTop: 0.1)
        railVertices.append(contentsOf: top.0)
        railIndices.append(contentsOf: top.1.map { $0 + current })
        current += UInt16(top.0.count)

        let bottom = buildRailBox(minX: -1.0, maxX: 1.0, minZ: 2.0, maxZ: 2.1, yBottom: 0.0, yTop: 0.1)
        railVertices.append(contentsOf: bottom.0)
        railIndices.append(contentsOf: bottom.1.map { $0 + current })
        current += UInt16(bottom.0.count)

        return (railVertices, railIndices)
    }

    private func buildPockets(startIndex: UInt16) -> ([Vertex], [UInt16]) {
        var pocketVerts: [Vertex] = []
        var pocketInds:  [UInt16] = []
        var curr = startIndex
        
        func buildPocket(at x: Float, z: Float) -> ([Vertex], [UInt16]) {
            let size: Float = 0.07
            let half = size * 0.5
            let y: Float = 0
            
            let p1 = SIMD3<Float>(x - half, y, z - half)
            let p2 = SIMD3<Float>(x + half, y, z - half)
            let p3 = SIMD3<Float>(x + half, y, z + half)
            let p4 = SIMD3<Float>(x - half, y, z + half)
            let n  = SIMD3<Float>(0,1,0)
            
            let uv0 = SIMD2<Float>(0,0)
            let uv1 = SIMD2<Float>(1,0)
            let uv2 = SIMD2<Float>(1,1)
            let uv3 = SIMD2<Float>(0,1)
            
            let vs = [
                Vertex(position: p1, normal: n, uv: uv0),
                Vertex(position: p2, normal: n, uv: uv1),
                Vertex(position: p3, normal: n, uv: uv2),
                Vertex(position: p4, normal: n, uv: uv3),
            ]
            let inds: [UInt16] = [0,1,2, 0,2,3]
            return (vs, inds)
        }
        
        let corners: [SIMD2<Float>] = [
            [-1, -2], [ 1, -2],
            [-1,  2], [ 1,  2]
        ]
        for c in corners {
            let (v, i) = buildPocket(at: c.x, z: c.y)
            pocketVerts.append(contentsOf: v)
            pocketInds.append(contentsOf: i.map { $0 + curr })
            curr += UInt16(v.count)
        }
        
        // sides
        let mids = [ SIMD2<Float>(-1, 0),
                     SIMD2<Float>( 1, 0) ]
        for m in mids {
            let (v, i) = buildPocket(at: m.x, z: m.y)
            pocketVerts.append(contentsOf: v)
            pocketInds.append(contentsOf: i.map { $0 + curr })
            curr += UInt16(v.count)
        }
        
        return (pocketVerts, pocketInds)
    }
}

//==================================================
// MARK: - SphereGeometry (for realistic balls)
//==================================================
struct SphereGeometry {
    var vertices: [Vertex] = []
    var indices:  [UInt16] = []
    
    init(radius: Float = 1.0, slices: Int = 12, stacks: Int = 12) {
        var tempVerts: [Vertex] = []
        var tempInds:  [UInt16] = []
        
        for stack in 0...stacks {
            let phi = Float(stack) / Float(stacks) * .pi
            for slice in 0...slices {
                let theta = Float(slice) / Float(slices) * 2.0 * .pi
                let x = radius * sin(phi) * cos(theta)
                let y = radius * cos(phi)
                let z = radius * sin(phi) * sin(theta)
                
                let nx = x / radius
                let ny = y / radius
                let nz = z / radius
                let u  = Float(slice) / Float(slices)
                let v  = Float(stack) / Float(stacks)
                
                tempVerts.append(
                    Vertex(position: SIMD3<Float>(x,y,z),
                           normal:   SIMD3<Float>(nx, ny, nz),
                           uv:       SIMD2<Float>(u, v))
                )
            }
        }
        
        let vertsPerRow = slices + 1
        for stack in 0..<stacks {
            for slice in 0..<slices {
                let i0 = UInt16(stack * vertsPerRow + slice)
                let i1 = UInt16((stack+1) * vertsPerRow + slice)
                let i2 = UInt16(i0 + 1)
                let i3 = UInt16(i1 + 1)
                
                tempInds.append(contentsOf: [i0, i1, i2, i2, i1, i3])
            }
        }
        
        vertices = tempVerts
        indices  = tempInds
    }
}

//==================================================
// MARK: - Renderer
//==================================================
final class Renderer: NSObject, MTKViewDelegate, ObservableObject {
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState
    private let depthState:   MTLDepthStencilState
    
    // Camera, table, sphere geometry
    var camera = Camera()
    private let table = BilliardTable()
    private let ballGeometry = SphereGeometry(radius: 1.0, slices: 12, stacks: 12)
    
    // Buffers for table geometry
    private var tableVertexBuffer: MTLBuffer?
    private var tableIndexBuffer:  MTLBuffer?
    
    // Buffers for sphere geometry
    private var sphereVertexBuffer: MTLBuffer?
    private var sphereIndexBuffer:  MTLBuffer?
    
    // 2 textures: felt & wood
    private var feltTexture: MTLTexture?
    private var woodTexture: MTLTexture?
    
    // Example ball positions
    private var balls: [SIMD3<Float>] = [
        [0, 0.05, 0],
        [0.5, 0.05, 0.3]
    ]
    
    init(metalKitView: MTKView) {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not supported.")
        }
        device = dev
        
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue.")
        }
        commandQueue = queue
        
        // Build library
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: metalShaderSource, options: nil)
        } catch {
            fatalError("Could not create library: \(error)")
        }
        
        // Pipeline
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = library.makeFunction(name: "vertexShader")
        desc.fragmentFunction = library.makeFunction(name: "fragmentShader")
        desc.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        desc.depthAttachmentPixelFormat      = .depth32Float
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format      = .float3
        vertexDescriptor.attributes[0].offset      = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[1].format      = .float3
        vertexDescriptor.attributes[1].offset      = MemoryLayout<SIMD3<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.attributes[2].format      = .float2
        vertexDescriptor.attributes[2].offset      = MemoryLayout<SIMD3<Float>>.stride * 2
        vertexDescriptor.attributes[2].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
        
        desc.vertexDescriptor = vertexDescriptor
        
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Could not create pipeline state: \(error)")
        }
        
        // Depth
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.isDepthWriteEnabled  = true
        depthDesc.depthCompareFunction = .less
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else {
            fatalError("Could not create depth state.")
        }
        depthState = ds
        
        super.init()
        
        // Setup geometry & textures
        setupTableBuffers()
        setupSphereBuffers()
        loadTextures()
    }
    
    private func setupTableBuffers() {
        tableVertexBuffer = device.makeBuffer(bytes: table.vertices,
                                              length: table.vertices.count * MemoryLayout<Vertex>.stride,
                                              options: [])
        tableIndexBuffer  = device.makeBuffer(bytes: table.indices,
                                              length: table.indices.count * MemoryLayout<UInt16>.stride,
                                              options: [])
    }
    
    private func setupSphereBuffers() {
        sphereVertexBuffer = device.makeBuffer(bytes: ballGeometry.vertices,
                                               length: ballGeometry.vertices.count * MemoryLayout<Vertex>.stride,
                                               options: [])
        sphereIndexBuffer  = device.makeBuffer(bytes: ballGeometry.indices,
                                               length: ballGeometry.indices.count * MemoryLayout<UInt16>.stride,
                                               options: [])
    }
    
    private func loadTextures() {
        let texLoader = MTKTextureLoader(device: device)
        
        // feltTexture.png must exist in your bundle
        if let feltURL = Bundle.main.url(forResource: "feltTexture", withExtension: "png") {
            feltTexture = try? texLoader.newTexture(URL: feltURL, options: [
                MTKTextureLoader.Option.SRGB : false
            ])
        }
        
        // woodTexture.png must exist in your bundle
        if let woodURL = Bundle.main.url(forResource: "woodTexture", withExtension: "png") {
            woodTexture = try? texLoader.newTexture(URL: woodURL, options: [
                MTKTextureLoader.Option.SRGB : false
            ])
        }
    }
    
    // Camera updates
    func updateCameraDrag(deltaX: Float, deltaY: Float) {
        camera.updateCameraDrag(deltaX: deltaX, deltaY: deltaY)
    }
    func updateCameraZoom(pinch: Float) {
        camera.updateCameraZoom(pinchScale: pinch)
    }
    
    // MTKViewDelegate
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspectRatio = Float(size.width / size.height)
    }
    
    func draw(in view: MTKView) {
        guard let pass = view.currentRenderPassDescriptor,
              let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            return
        }
        
        encoder.setRenderPipelineState(pipelineState)
        encoder.setDepthStencilState(depthState)
        
        // Sampler
        var samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter = .linear
        samplerDesc.magFilter = .linear
        samplerDesc.mipFilter = .linear
        samplerDesc.sAddressMode = .repeat
        samplerDesc.tAddressMode = .repeat
        let sampler = device.makeSamplerState(descriptor: samplerDesc)!
        
        // Set fragment textures
        if let felt = feltTexture {
            encoder.setFragmentTexture(felt, index: 0)
        }
        if let wood = woodTexture {
            encoder.setFragmentTexture(wood, index: 1)
        }
        encoder.setFragmentSamplerState(sampler, index: 0)
        
        // 1) Table felt
        drawTableFelt(with: encoder)
        
        // 2) Table rails/pockets
        drawTableWood(with: encoder)
        
        // 3) Billiard balls
        drawBalls(with: encoder)
        
        encoder.endEncoding()
        if let drawable = view.currentDrawable {
            cmdBuffer.present(drawable)
        }
        cmdBuffer.commit()
    }
    
    private func drawTableFelt(with encoder: MTLRenderCommandEncoder) {
        guard let vb = tableVertexBuffer, let ib = tableIndexBuffer else { return }
        let feltIndexCount = 6
        
        var uniforms = Uniforms(
            modelMatrix:      table.modelMatrix,
            viewMatrix:       camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            lightPosition:    [0,2,2]
        )
        var materialId: Float = 0 // 0 => felt
        
        encoder.setVertexBuffer(vb, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setVertexBytes(&materialId, length: MemoryLayout<Float>.stride, index: 2)
        
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: feltIndexCount,
                                      indexType: .uint16,
                                      indexBuffer: ib,
                                      indexBufferOffset: 0)
    }
    
    private func drawTableWood(with encoder: MTLRenderCommandEncoder) {
        guard let vb = tableVertexBuffer, let ib = tableIndexBuffer else { return }
        
        let feltIndexCount = 6
        let totalIndexCount = table.indices.count
        let woodIndexCount  = totalIndexCount - feltIndexCount
        let woodOffset      = feltIndexCount * MemoryLayout<UInt16>.stride
        
        var uniforms = Uniforms(
            modelMatrix:      table.modelMatrix,
            viewMatrix:       camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            lightPosition:    [0,2,2]
        )
        var materialId: Float = 1 // 1 => wood
        
        encoder.setVertexBuffer(vb, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setVertexBytes(&materialId, length: MemoryLayout<Float>.stride, index: 2)
        
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: woodIndexCount,
                                      indexType: .uint16,
                                      indexBuffer: ib,
                                      indexBufferOffset: woodOffset)
    }
    
    private func drawBalls(with encoder: MTLRenderCommandEncoder) {
        guard let vb = sphereVertexBuffer, let ib = sphereIndexBuffer else { return }
        let indexCount = ballGeometry.indices.count
        
        encoder.setVertexBuffer(vb, offset: 0, index: 0)
        
        for pos in balls {
            // We'll pass materialId=10 => fallback color
            var materialId: Float = 10
            
            let scaleMat = scale(0.05)
            let transMat = translate(pos)
            let model = transMat * scaleMat
            
            var uniforms = Uniforms(
                modelMatrix:      model,
                viewMatrix:       camera.viewMatrix,
                projectionMatrix: camera.projectionMatrix,
                lightPosition:    [0,2,2]
            )
            
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            encoder.setVertexBytes(&materialId, length: MemoryLayout<Float>.stride, index: 2)
            
            encoder.drawIndexedPrimitives(type: .triangle,
                                          indexCount: indexCount,
                                          indexType: .uint16,
                                          indexBuffer: ib,
                                          indexBufferOffset: 0)
        }
    }
    
    private func translate(_ pos: SIMD3<Float>) -> matrix_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4<Float>(pos, 1)
        return m
    }
    
    private func scale(_ s: Float) -> matrix_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.0.x = s
        m.columns.1.y = s
        m.columns.2.z = s
        return m
    }
}

//==================================================
// MARK: - SwiftUI MetalView
//==================================================
struct MetalView: UIViewRepresentable {
    @ObservedObject var renderer: Renderer
    
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: renderer.device)
        view.delegate = renderer
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColorMake(0.2, 0.3, 0.4, 1)
        return view
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {}
}

//==================================================
// MARK: - ContentView
//==================================================
struct ContentView: View {
    @StateObject private var renderer: Renderer
    
    @State private var lastDrag: CGSize  = .zero
    @State private var lastPinch: CGFloat = 1.0
    
    init() {
        let tempView = MTKView()
        let tempRenderer = Renderer(metalKitView: tempView)
        _renderer = StateObject(wrappedValue: tempRenderer)
    }
    
    var body: some View {
        GeometryReader { geo in
            ZStack {
                MetalView(renderer: renderer)
                    .gesture(
                        // Drag => rotate/pitch
                        DragGesture()
                            .onChanged { value in
                                let dx = Float(value.translation.width  - lastDrag.width)
                                let dy = Float(value.translation.height - lastDrag.height)
                                renderer.updateCameraDrag(deltaX: dx, deltaY: dy)
                                lastDrag = value.translation
                            }
                            .onEnded { _ in
                                lastDrag = .zero
                            }
                    )
                    .gesture(
                        // Pinch => zoom
                        MagnificationGesture()
                            .onChanged { val in
                                let pinchDelta = val - lastPinch
                                renderer.updateCameraZoom(pinch: Float(pinchDelta))
                                lastPinch = val
                            }
                            .onEnded { _ in
                                lastPinch = 1.0
                            }
                    )
            }
            .onAppear {
                // Adjust camera aspect ratio
                renderer.camera.aspectRatio = Float(geo.size.width / geo.size.height)
            }
        }
    }
}

//==================================================
// MARK: - App Entry
//==================================================
@main
struct BilliardsApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
