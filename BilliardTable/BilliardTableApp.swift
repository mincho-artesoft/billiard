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
                               texture2d<float> woodTex  [[texture(0)]],
                               sampler samp [[sampler(0)]])
{
    float3 normal   = normalize(inVertex.normal);
    // Light direction from above (0,6,6) or passed in from Swift. 
    // For demonstration, we do a quick approximate:
    float3 lightDir = normalize(inVertex.worldPos - float3(0.0, -6.0, -6.0)); 
    // Dot product for Lambertian
    float NdotL     = max(dot(normal, -lightDir), 0.0);
    
    float3 textureColor;
    // (1) 0.0 => felt => bright green
    if (inVertex.materialId < 0.5) {
        textureColor = float3(0.0, 1.0, 0.0);
    }
    // (2) 0.5 <= ID < 1.5 => wood => sample wood texture
    else if (inVertex.materialId < 1.5) {
        float4 c = woodTex.sample(samp, inVertex.uv);
        textureColor = c.rgb;
    }
    // (3) 1.5 <= ID < 2.5 => pockets => black
    else if (inVertex.materialId < 2.5) {
        textureColor = float3(0.0, 0.0, 0.0);
    }
    // (4) fallback => red (e.g., balls)
    else {
        textureColor = float3(1.0, 0.0, 0.0);
    }
    
    // Increase ambient and keep a diffuse term
    float3 ambient = 0.4 * textureColor;
    float3 diffuse = 0.6 * textureColor * NdotL;
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
        // clamp pitch
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
        perspectiveFov(fovy: fov, aspect: aspectRatio, nearZ: 0.01, farZ: 100)
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
        let (feltVerts, feltInds)     = buildFeltSurface()
        let (railVerts, railInds)     = buildRails(startIndex: UInt16(feltVerts.count))
        let (pockVerts, pockInds)     = buildPocketCylinders(startIndex: UInt16(feltVerts.count + railVerts.count))
        
        vertices = feltVerts + railVerts + pockVerts
        indices  = feltInds  + railInds  + pockInds
    }

    // MARK: - Felt (green) at y=0
    private func buildFeltSurface() -> ([Vertex], [UInt16]) {
        let p1: SIMD3<Float> = [-1, 0, -2]
        let p2: SIMD3<Float> = [ 1, 0, -2]
        let p3: SIMD3<Float> = [ 1, 0,  2]
        let p4: SIMD3<Float> = [-1, 0,  2]
        
        let n: SIMD3<Float>  = [0, 1, 0]
        let uv1: SIMD2<Float> = [0,0]
        let uv2: SIMD2<Float> = [1,0]
        let uv3: SIMD2<Float> = [1,1]
        let uv4: SIMD2<Float> = [0,1]
        
        let verts = [
            Vertex(position: p1, normal: n, uv: uv1),
            Vertex(position: p2, normal: n, uv: uv2),
            Vertex(position: p3, normal: n, uv: uv3),
            Vertex(position: p4, normal: n, uv: uv4),
        ]
        let inds: [UInt16] = [0,1,2, 0,2,3]
        return (verts, inds)
    }

    // MARK: - Wood Rails (Split into smaller segments to leave pockets open)
    private func buildRails(startIndex: UInt16) -> ([Vertex], [UInt16]) {
        var railVertices: [Vertex] = []
        var railIndices:  [UInt16] = []
        var current = startIndex
        
        let railMinY: Float = 0.0
        let railMaxY: Float = 0.1
        
        let cornerMarginX: Float = 0.06  // Reduced from 0.12 as per your previous request
        let cornerMarginZ: Float = 0.06  // Reduced from 0.12 as per your previous request
        
        let railOuterLeftX:  Float = -1.1
        let railInnerLeftX:  Float = -1.0
        let railOuterRightX: Float =  1.1
        let railInnerRightX: Float =  1.0
        
        let railOuterTopZ:   Float = -2.1
        let railInnerTopZ:   Float = -2.0
        let railOuterBotZ:   Float =  2.1
        let railInnerBotZ:   Float =  2.0
        
        // Helper to build one rectangular "box" segment
        func buildRailBox(minX: Float, maxX: Float,
                          minZ: Float, maxZ: Float,
                          yBottom: Float, yTop: Float) -> ([Vertex], [UInt16])
        {
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
            
            let n  = SIMD3<Float>(0,1,0)
            let uv = SIMD2<Float>(0,0)
            
            let vs = corners.map { Vertex(position: $0, normal: n, uv: uv) }
            let iset: [UInt16] = [
                0,1,2, 0,2,3,    // bottom
                4,5,6, 4,6,7,    // top
                0,1,5, 0,5,4,    // front
                2,3,7, 2,7,6,    // back
                0,3,7, 0,7,4,    // left
                1,2,6, 1,6,5     // right
            ]
            return (vs, iset)
        }
        
        // LEFT RAIL: two segments, leaving corners & mid side open
        let left1 = buildRailBox(
            minX:  railOuterLeftX, maxX:  railInnerLeftX,
            minZ: -2.0 + cornerMarginZ, maxZ: -cornerMarginZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: left1.0)
        railIndices.append(contentsOf: left1.1.map { $0 + current })
        current += UInt16(left1.0.count)
        
        let left2 = buildRailBox(
            minX:  railOuterLeftX, maxX:  railInnerLeftX,
            minZ:  cornerMarginZ, maxZ:  2.0 - cornerMarginZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: left2.0)
        railIndices.append(contentsOf: left2.1.map { $0 + current })
        current += UInt16(left2.0.count)
        
        // RIGHT RAIL: two segments
        let right1 = buildRailBox(
            minX:  railInnerRightX, maxX:  railOuterRightX,
            minZ: -2.0 + cornerMarginZ, maxZ: -cornerMarginZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: right1.0)
        railIndices.append(contentsOf: right1.1.map { $0 + current })
        current += UInt16(right1.0.count)
        
        let right2 = buildRailBox(
            minX:  railInnerRightX, maxX:  railOuterRightX,
            minZ:  cornerMarginZ, maxZ:  2.0 - cornerMarginZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: right2.0)
        railIndices.append(contentsOf: right2.1.map { $0 + current })
        current += UInt16(right2.0.count)
        
        // TOP RAIL: single continuous segment (merged top1 and top2)
        let top = buildRailBox(
            minX: -1.0 + cornerMarginX, maxX: 1.0 - cornerMarginX,  // Full length without middle gap
            minZ:  railOuterTopZ, maxZ:  railInnerTopZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: top.0)
        railIndices.append(contentsOf: top.1.map { $0 + current })
        current += UInt16(top.0.count)
        
        // BOTTOM RAIL: single continuous segment (merged bot1 and bot2)
        let bottom = buildRailBox(
            minX: -1.0 + cornerMarginX, maxX: 1.0 - cornerMarginX,  // Full length without middle gap
            minZ:  railInnerBotZ, maxZ:  railOuterBotZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: bottom.0)
        railIndices.append(contentsOf: bottom.1.map { $0 + current })
        current += UInt16(bottom.0.count)
        
        return (railVertices, railIndices)
    }
    // MARK: - Cylindrical Pockets (materialId=2.0 => black)
    private func buildPocketCylinders(startIndex: UInt16) -> ([Vertex], [UInt16]) {
        var pocketVerts: [Vertex] = []
        var pocketInds:  [UInt16] = []
        var current = startIndex
        
        
        
        // Helper to build an open cylinder along the y-axis:
        // Top ring at y=0, bottom ring at y=-height
        func makeOpenCylinder(radius: Float, height: Float, radialSegments: Int) -> ([Vertex], [UInt16]) {
            var verts: [Vertex] = []
            var inds:  [UInt16] = []
            
            for i in 0...radialSegments {
                let theta = Float(i) / Float(radialSegments) * 2.0 * Float.pi
                let x = radius * cos(theta)
                let z = radius * sin(theta)
                
                // Top ring vertex (y=0)
                let topPos = SIMD3<Float>(x, 0, z)
                // Outward normal => from table center outward
                let topNrm = simd_normalize(SIMD3<Float>(x, 0, z))
                
                // Bottom ring vertex (y=-height)
                let botPos = SIMD3<Float>(x, -height, z)
                let botNrm = topNrm
                
                verts.append(Vertex(position: topPos, normal: topNrm, uv: SIMD2<Float>(0, 0)))
                verts.append(Vertex(position: botPos, normal: botNrm, uv: SIMD2<Float>(1, 1)))
            }
            
            // Connect the rings with quads:
            let stride = 2
            for i in 0..<radialSegments {
                let iTop0 = UInt16(i * stride)
                let iBot0 = iTop0 + 1
                let iTop1 = iTop0 + UInt16(stride)
                let iBot1 = iTop1 + 1
                
                // (top0, bot0, top1), (top1, bot0, bot1)
                inds.append(iTop0)
                inds.append(iBot0)
                inds.append(iTop1)
                inds.append(iTop1)
                inds.append(iBot0)
                inds.append(iBot1)
            }
            return (verts, inds)
        }
        
        // Build pockets at corners + left/right midpoints
        let radius: Float = 0.06  // Reduced from 0.08
        let depth:  Float = 0.15
        
        func buildPocket(at x: Float, z: Float, radius: Float, depth: Float) -> ([Vertex], [UInt16]) {
            let (localVerts, localInds) = makeOpenCylinder(radius: radius, height: depth, radialSegments: 12)
            let transform = translate(x, 0, z)
            
            // Transform each vertex
            let transformedVerts = localVerts.map { v -> Vertex in
                let pos4  = SIMD4<Float>(v.position, 1)
                let newPos = transform * pos4
                let nrm4  = transform * SIMD4<Float>(v.normal, 0)
                
                return Vertex(position: SIMD3<Float>(newPos.x, newPos.y, newPos.z),
                              normal:   simd_normalize(SIMD3<Float>(nrm4.x, nrm4.y, nrm4.z)),
                              uv:       v.uv)
            }
            return (transformedVerts, localInds)
        }
        
        // Corner pockets
        let corners: [SIMD2<Float>] = [
            [-1, -2], [ 1, -2],
            [-1,  2], [ 1,  2]
        ]
        for c in corners {
            let (v, i) = buildPocket(at: c.x, z: c.y, radius: radius, depth: depth)
            pocketVerts.append(contentsOf: v)
            pocketInds.append(contentsOf: i.map { $0 + current })
            current += UInt16(v.count)
        }
        
        // Side pockets (left/right midpoints)
        let mids = [
            SIMD2<Float>(-1, 0),
            SIMD2<Float>( 1, 0)
        ]
        for m in mids {
            let (v, i) = buildPocket(at: m.x, z: m.y, radius: radius, depth: depth)
            pocketVerts.append(contentsOf: v)
            pocketInds.append(contentsOf: i.map { $0 + current })
            current += UInt16(v.count)
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
    private var sphereIndexBuffer: MTLBuffer?
    
    // Texture for rails
    private var woodTexture: MTLTexture?
    
    // Example ball positions
    private var balls: [SIMD3<Float>] = [
        [0, 0.05, 0],
        [0.5, 0.05, 0.3]
    ]
    
    init(metalKitView: MTKView) {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not supported on this device.")
        }
        device = dev
        
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue.")
        }
        commandQueue = queue
        
        // Build library from embedded shader source
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: metalShaderSource, options: nil)
        } catch {
            fatalError("Could not create library: \(error)")
        }
        
        // Create pipeline
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = library.makeFunction(name: "vertexShader")
        desc.fragmentFunction = library.makeFunction(name: "fragmentShader")
        
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.depthAttachmentPixelFormat      = .depth32Float
        
        // Vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format      = .float3
        vertexDescriptor.attributes[0].offset      = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        vertexDescriptor.attributes[1].format      = .float3
        vertexDescriptor.attributes[1].offset      = MemoryLayout<SIMD3<Float>>.stride
        vertexDescriptor.attributes[2].format      = .float2
        vertexDescriptor.attributes[2].offset      = MemoryLayout<SIMD3<Float>>.stride * 2
        
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.attributes[2].bufferIndex = 0
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
        
        desc.vertexDescriptor = vertexDescriptor
        
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Could not create pipeline state: \(error)")
        }
        
        // Depth state
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.isDepthWriteEnabled  = true
        depthDesc.depthCompareFunction = .less
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else {
            fatalError("Could not create depth state.")
        }
        depthState = ds
        
        super.init()
        
        // Setup geometry & texture
        setupTableBuffers()
        setupSphereBuffers()
        loadTextures()
    }
    
    private func setupTableBuffers() {
        tableVertexBuffer = device.makeBuffer(
            bytes: table.vertices,
            length: table.vertices.count * MemoryLayout<Vertex>.stride,
            options: []
        )
        tableIndexBuffer  = device.makeBuffer(
            bytes: table.indices,
            length: table.indices.count * MemoryLayout<UInt16>.stride,
            options: []
        )
    }
    
    private func setupSphereBuffers() {
        sphereVertexBuffer = device.makeBuffer(
            bytes: ballGeometry.vertices,
            length: ballGeometry.vertices.count * MemoryLayout<Vertex>.stride,
            options: []
        )
        sphereIndexBuffer  = device.makeBuffer(
            bytes: ballGeometry.indices,
            length: ballGeometry.indices.count * MemoryLayout<UInt16>.stride,
            options: []
        )
    }
    
    private func loadTextures() {
        let texLoader = MTKTextureLoader(device: device)
        
        // Load wood texture for rails
        if let woodURL = Bundle.main.url(forResource: "woodTexture", withExtension: "png") {
            woodTexture = try? texLoader.newTexture(
                URL: woodURL,
                options: [ MTKTextureLoader.Option.SRGB : false ]
            )
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
        
        // Basic sampler for the wood texture
        var samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter    = .linear
        samplerDesc.magFilter    = .linear
        samplerDesc.mipFilter    = .linear
        samplerDesc.sAddressMode = .repeat
        samplerDesc.tAddressMode = .repeat
        
        guard let sampler = device.makeSamplerState(descriptor: samplerDesc) else { return }
        if let wood = woodTexture {
            encoder.setFragmentTexture(wood, index: 0)
        }
        encoder.setFragmentSamplerState(sampler, index: 0)
        
        // 1) Table felt
        drawTableFelt(with: encoder)
        
        // 2) Wood rails + pockets
        drawTableWoodAndPockets(with: encoder)
        
        // 3) Balls
        drawBalls(with: encoder)
        
        encoder.endEncoding()
        if let drawable = view.currentDrawable {
            cmdBuffer.present(drawable)
        }
        cmdBuffer.commit()
    }
    
    private func drawTableFelt(with encoder: MTLRenderCommandEncoder) {
        guard let vb = tableVertexBuffer,
              let ib = tableIndexBuffer else { return }
        
        // The felt uses the first 6 indices
        let feltIndexCount = 6
        
        var uniforms = Uniforms(
            modelMatrix:      table.modelMatrix,
            viewMatrix:       camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            lightPosition:    [0, 6, 6]
        )
        // materialId=0 => green
        var materialId: Float = 0.0
        
        encoder.setVertexBuffer(vb, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setVertexBytes(&materialId, length: MemoryLayout<Float>.stride, index: 2)
        
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: feltIndexCount,
                                      indexType: .uint16,
                                      indexBuffer: ib,
                                      indexBufferOffset: 0)
    }
    
    private func drawTableWoodAndPockets(with encoder: MTLRenderCommandEncoder) {
        guard let vb = tableVertexBuffer,
              let ib = tableIndexBuffer else { return }
        
        let feltIndexCount  = 6
        let totalIndexCount = table.indices.count
        let woodAndPocketIndexCount = totalIndexCount - feltIndexCount
        let woodPocketOffsetBytes   = feltIndexCount * MemoryLayout<UInt16>.stride
        
        var uniforms = Uniforms(
            modelMatrix:      table.modelMatrix,
            viewMatrix:       camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            lightPosition:    [0, 6, 6]
        )
        // For simplicity, we set materialId=1.0 => “wood,” but
        // the pockets remain black because of the ID check in the fragment shader.
        var materialId: Float = 1.0
        
        encoder.setVertexBuffer(vb, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setVertexBytes(&materialId, length: MemoryLayout<Float>.stride, index: 2)
        
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: woodAndPocketIndexCount,
                                      indexType: .uint16,
                                      indexBuffer: ib,
                                      indexBufferOffset: woodPocketOffsetBytes)
    }
    
    private func drawBalls(with encoder: MTLRenderCommandEncoder) {
        guard let vb = sphereVertexBuffer,
              let ib = sphereIndexBuffer else { return }
        
        let indexCount = ballGeometry.indices.count
        
        encoder.setVertexBuffer(vb, offset: 0, index: 0)
        
        for pos in balls {
            // materialId=10 => fallback => red
            var materialId: Float = 10.0
            
            let scaleMat = scale(0.05)  // ~5 cm radius
            let transMat = translate(pos.x, pos.y, pos.z)
            let model    = transMat * scaleMat
            
            var uniforms = Uniforms(
                modelMatrix:      model,
                viewMatrix:       camera.viewMatrix,
                projectionMatrix: camera.projectionMatrix,
                lightPosition:    [0, 6, 6]
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
}

//==================================================
// MARK: - SwiftUI MetalView
//==================================================
struct MetalView: UIViewRepresentable {
    @ObservedObject var renderer: Renderer
    
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: renderer.device)
        view.colorPixelFormat        = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColorMake(0.8, 0.9, 0.8, 1.0)
        
        view.delegate = renderer
        return view
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {}
}

//==================================================
// MARK: - ContentView
//==================================================
struct ContentView: View {
    @StateObject private var renderer: Renderer
    
    @State private var lastDrag: CGSize   = .zero
    @State private var lastPinch: CGFloat = 1.0
    
    init() {
        // Create a temporary MTKView to initialize the renderer
        let tempView = MTKView()
        tempView.colorPixelFormat = .bgra8Unorm
        
        let tempRenderer = Renderer(metalKitView: tempView)
        _renderer = StateObject(wrappedValue: tempRenderer)
    }
    
    var body: some View {
        GeometryReader { geo in
            ZStack {
                MetalView(renderer: renderer)
                    .gesture(
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
                renderer.camera.aspectRatio = Float(geo.size.width / geo.size.height)
            }
        }
    }
}

//==================================================
// MARK: - Transform Helpers
//==================================================
fileprivate func translate(_ x: Float, _ y: Float, _ z: Float) -> matrix_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.3 = SIMD4<Float>(x, y, z, 1)
    return m
}

fileprivate func translate(_ pos: SIMD3<Float>) -> matrix_float4x4 {
    translate(pos.x, pos.y, pos.z)
}

fileprivate func scale(_ s: Float) -> matrix_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.0.x = s
    m.columns.1.y = s
    m.columns.2.z = s
    return m
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
