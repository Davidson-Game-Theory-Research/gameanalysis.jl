
function in_simplex(x,y)
    y ≥ 0 && y ≤ √3x && y ≤ √3(1-x)
end

function from_barycentric(λ₁, λ₂, λ₃)
    r₁ = [0 0]
    r₂ = [1 0]
    r₃ = [.5 √3]
    λ₁*r₁ + λ₂*r₂ + λ₃*r₃
end

function to_barycentric(x, y)
    x₁,y₁ = [0 0]
    x₂,y₂ = [1 0]
    x₃,y₃ = [.5 √3]
    λ₁ = ((y₂-y₃)*(x-x₃) + (x₃-x₂)*(y-y₃)) / ((y₂-y₃)*(x₁-x₃) + (x₃-x₂)*(y₁-y₃))
    λ₂ = ((y₃-y₁)*(x-x₃) + (x₁-x₃)*(y-y₃)) / ((y₂-y₃)*(x₁-x₃) + (x₃-x₂)*(y₁-y₃))
    [λ₁ λ₂ 1-λ₁-λ₂]
end
