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
        let (feltVerts, feltInds) = buildFeltSurface()
        let (railVerts, railInds) = buildRails(startIndex: UInt16(feltVerts.count))
        
        vertices = feltVerts + railVerts
        indices  = feltInds  + railInds
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

    // MARK: - Wood Rails with Pocket Holes
    private func buildRails(startIndex: UInt16) -> ([Vertex], [UInt16]) {
        var railVertices: [Vertex] = []
        var railIndices:  [UInt16] = []
        var current = startIndex
        
        let railMinY: Float = 0.0
        let railMaxY: Float = 0.1
        
        let cornerMarginX: Float = 0.06
        let cornerMarginZ: Float = 0.06
        
        let railOuterLeftX:  Float = -1.1
        let railInnerLeftX:  Float = -1.0
        let railOuterRightX: Float =  1.1
        let railInnerRightX: Float =  1.0
        
        let railOuterTopZ:   Float = -2.1
        let railInnerTopZ:   Float = -2.0
        let railOuterBotZ:   Float =  2.1
        let railInnerBotZ:   Float =  2.0
        
        // Pocket positions and radius
        let pocketRadius: Float = 0.06
        let pocketCenters: [SIMD2<Float>] = [
            [-1, -2], [1, -2], [-1, 2], [1, 2],  // Corners
            [-1, 0], [1, 0]                      // Side midpoints
        ]

        // Helper to check if a point is inside any pocket
        func isInPocket(x: Float, z: Float) -> Bool {
            for center in pocketCenters {
                let dx = x - center.x
                let dz = z - center.y
                let distance = sqrt(dx * dx + dz * dz)
                if distance < pocketRadius {
                    return true
                }
            }
            return false
        }

        // Helper to build a rail box with pocket holes
        func buildRailBox(minX: Float, maxX: Float,
                          minZ: Float, maxZ: Float,
                          yBottom: Float, yTop: Float) -> ([Vertex], [UInt16])
        {
            var verts: [Vertex] = []
            var inds:  [UInt16] = []
            var vertexMap: [SIMD3<Float>: UInt16] = [:]

            // Define grid resolution for rail surface
            let segmentsX = 20
            let segmentsZ = Int((maxZ - minZ) / (maxX - minX) * Float(segmentsX))

            // Generate vertices for top face only (since pockets are holes)
            for i in 0...segmentsX {
                let x = minX + (maxX - minX) * Float(i) / Float(segmentsX)
                for j in 0...segmentsZ {
                    let z = minZ + (maxZ - minZ) * Float(j) / Float(segmentsZ)
                    let posBottom = SIMD3<Float>(x, yBottom, z)
                    let posTop = SIMD3<Float>(x, yTop, z)

                    // Normals for each face
                    let nTop = SIMD3<Float>(0, 1, 0)
                    let nBottom = SIMD3<Float>(0, -1, 0)
                    let nLeft = SIMD3<Float>(-1, 0, 0)
                    let nRight = SIMD3<Float>(1, 0, 0)
                    let nFront = SIMD3<Float>(0, 0, -1)
                    let nBack = SIMD3<Float>(0, 0, 1)
                    let uv = SIMD2<Float>(x / (maxX - minX), z / (maxZ - minZ))

                    // Add bottom face vertex
                    if !isInPocket(x: x, z: z) {
                        if vertexMap[posBottom] == nil {
                            vertexMap[posBottom] = UInt16(verts.count)
                            verts.append(Vertex(position: posBottom, normal: nBottom, uv: uv))
                        }
                    }

                    // Add top face vertex (with pocket holes)
                    if !isInPocket(x: x, z: z) {
                        if vertexMap[posTop] == nil {
                            vertexMap[posTop] = UInt16(verts.count)
                            verts.append(Vertex(position: posTop, normal: nTop, uv: uv))
                        }
                    }
                }
            }

            // Generate indices for top face with holes
            for i in 0..<segmentsX {
                for j in 0..<segmentsZ {
                    let x0 = minX + (maxX - minX) * Float(i) / Float(segmentsX)
                    let x1 = minX + (maxX - minX) * Float(i + 1) / Float(segmentsX)
                    let z0 = minZ + (maxZ - minZ) * Float(j) / Float(segmentsZ)
                    let z1 = minZ + (maxZ - minZ) * Float(j + 1) / Float(segmentsZ)

                    let top0 = SIMD3<Float>(x0, yTop, z0)
                    let top1 = SIMD3<Float>(x1, yTop, z0)
                    let top2 = SIMD3<Float>(x1, yTop, z1)
                    let top3 = SIMD3<Float>(x0, yTop, z1)

                    let bot0 = SIMD3<Float>(x0, yBottom, z0)
                    let bot1 = SIMD3<Float>(x1, yBottom, z0)
                    let bot2 = SIMD3<Float>(x1, yBottom, z1)
                    let bot3 = SIMD3<Float>(x0, yBottom, z1)

                    // Check if quad is fully or partially inside a pocket
                    let topCorners = [top0, top1, top2, top3]
                    let botCorners = [bot0, bot1, bot2, bot3]
                    let allTopInPocket = topCorners.allSatisfy { isInPocket(x: $0.x, z: $0.z) }
                    let allBotInPocket = botCorners.allSatisfy { isInPocket(x: $0.x, z: $0.z) }

                    if !allTopInPocket {
                        if let i0 = vertexMap[top0], let i1 = vertexMap[top1], let i2 = vertexMap[top2], let i3 = vertexMap[top3] {
                            inds.append(i0)
                            inds.append(i1)
                            inds.append(i2)
                            inds.append(i0)
                            inds.append(i2)
                            inds.append(i3)
                        }
                    }

                    if !allBotInPocket {
                        if let i0 = vertexMap[bot0], let i1 = vertexMap[bot1], let i2 = vertexMap[bot2], let i3 = vertexMap[bot3] {
                            inds.append(i0)
                            inds.append(i2)
                            inds.append(i1)
                            inds.append(i0)
                            inds.append(i3)
                            inds.append(i2)
                        }
                    }

                    // Side faces (front, back, left, right)
                    if i == 0 {
                        if let i0 = vertexMap[bot0], let i1 = vertexMap[top0], let i2 = vertexMap[top3], let i3 = vertexMap[bot3] {
                            if !isInPocket(x: x0, z: z0) && !isInPocket(x: x0, z: z1) {
                                inds.append(i0)
                                inds.append(i1)
                                inds.append(i2)
                                inds.append(i0)
                                inds.append(i2)
                                inds.append(i3)
                            }
                        }
                    }
                    if i == segmentsX - 1 {
                        if let i0 = vertexMap[bot1], let i1 = vertexMap[top1], let i2 = vertexMap[top2], let i3 = vertexMap[bot2] {
                            if !isInPocket(x: x1, z: z0) && !isInPocket(x: x1, z: z1) {
                                inds.append(i0)
                                inds.append(i2)
                                inds.append(i1)
                                inds.append(i0)
                                inds.append(i3)
                                inds.append(i2)
                            }
                        }
                    }
                    if j == 0 {
                        if let i0 = vertexMap[bot0], let i1 = vertexMap[top0], let i2 = vertexMap[top1], let i3 = vertexMap[bot1] {
                            if !isInPocket(x: x0, z: z0) && !isInPocket(x: x1, z: z0) {
                                inds.append(i0)
                                inds.append(i1)
                                inds.append(i2)
                                inds.append(i0)
                                inds.append(i2)
                                inds.append(i3)
                            }
                        }
                    }
                    if j == segmentsZ - 1 {
                        if let i0 = vertexMap[bot3], let i1 = vertexMap[top3], let i2 = vertexMap[top2], let i3 = vertexMap[bot2] {
                            if !isInPocket(x: x0, z: z1) && !isInPocket(x: x1, z: z1) {
                                inds.append(i0)
                                inds.append(i2)
                                inds.append(i1)
                                inds.append(i0)
                                inds.append(i3)
                                inds.append(i2)
                            }
                        }
                    }
                }
            }
            return (verts, inds)
        }

        // LEFT RAIL: Continuous from -2.0 + cornerMarginZ to 2.0 - cornerMarginZ
        let left = buildRailBox(
            minX: railOuterLeftX, maxX: railInnerLeftX,
            minZ: -2.0 + cornerMarginZ, maxZ: 2.0 - cornerMarginZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: left.0)
        railIndices.append(contentsOf: left.1.map { $0 + current })
        current += UInt16(left.0.count)

        // RIGHT RAIL: Continuous from -2.0 + cornerMarginZ to 2.0 - cornerMarginZ
        let right = buildRailBox(
            minX: railInnerRightX, maxX: railOuterRightX,
            minZ: -2.0 + cornerMarginZ, maxZ: 2.0 - cornerMarginZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: right.0)
        railIndices.append(contentsOf: right.1.map { $0 + current })
        current += UInt16(right.0.count)

        // TOP RAIL: Continuous from -1.0 + cornerMarginX to 1.0 - cornerMarginX
        let top = buildRailBox(
            minX: -1.0 + cornerMarginX, maxX: 1.0 - cornerMarginX,
            minZ: railOuterTopZ, maxZ: railInnerTopZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: top.0)
        railIndices.append(contentsOf: top.1.map { $0 + current })
        current += UInt16(top.0.count)

        // BOTTOM RAIL: Continuous from -1.0 + cornerMarginX to 1.0 - cornerMarginX
        let bottom = buildRailBox(
            minX: -1.0 + cornerMarginX, maxX: 1.0 - cornerMarginX,
            minZ: railInnerBotZ, maxZ: railOuterBotZ,
            yBottom: railMinY, yTop: railMaxY
        )
        railVertices.append(contentsOf: bottom.0)
        railIndices.append(contentsOf: bottom.1.map { $0 + current })
        current += UInt16(bottom.0.count)
        
        return (railVertices, railIndices)
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
        let woodIndexCount  = totalIndexCount - feltIndexCount
        let woodOffsetBytes = feltIndexCount * MemoryLayout<UInt16>.stride
        
        var uniforms = Uniforms(
            modelMatrix:      table.modelMatrix,
            viewMatrix:       camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            lightPosition:    [0, 6, 6]
        )
        var materialId: Float = 1.0  // Wood only, no separate pocket material
        
        encoder.setVertexBuffer(vb, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setVertexBytes(&materialId, length: MemoryLayout<Float>.stride, index: 2)
        
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: woodIndexCount,
                                      indexType: .uint16,
                                      indexBuffer: ib,
                                      indexBufferOffset: woodOffsetBytes)
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
