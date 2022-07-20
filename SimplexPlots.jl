using GMT
using LaTeXStrings

const bottom_annotation = L"@[\left< 0, \frac{1}{2}, \frac{1}{2}\right>@["
const left_annotation = L"@[\left< \frac{1}{2}, 0, \frac{1}{2}\right>@["
const right_annotation = L"@[\left< \frac{1}{2}, \frac{1}{2}, 0\right>@["

function simplex_setup(title=nothing; border=false, show=false, vertex_labels=["S1","S2","S3"])
    ternary(region=(0,1,0,1,0,1), frame=(grid=0, ticks=0, annot=0))
    simplex_setup!(title; border=border, show=show, vertex_labels=vertex_labels)
end

function simplex_setup!(title=nothing; border=false, show=false, vertex_labels=["S1","S2","S3"])
    if border
        plot!(tern2cart([0 0 1; 0 1 0; 1 0 0; 0 0 1]), S="~n1:+sc0", par=(MAP_DEFAULT_PEN="2,black",))
    end
    if !isnothing(title)
        text!([0.5 0.93], text=title, font=12)
    end
    text!([0.5 0.89], text=vertex_labels[1], font=10, noclip=true)
    text!([-0.02 -0.015], text=vertex_labels[2], noclip=true)
    text!([1.02 -0.015], text=vertex_labels[3], noclip=true)
    text!(tern2cart([-.05 0.48 .5]), text=right_annotation, font=7, noclip=true)
    text!(tern2cart([0.5 0.48 -.05]), text=left_annotation, font=7, noclip=true)
    text!(tern2cart([0.5 -.04 .5]), text=bottom_annotation, font=7, noclip=true)
    plot!(tern2cart([.003 .4985 .4985; .4985 .003 .4985; .4985 .4985 .003]), marker="^",
            markerfacecolor=:black, markersize=.2, noclip=true, show=show)
end

function simplex_heatmap(mixtures::Matrix, values::Array; title=nothing, show=false, vertex_labels=["S1","S2","S3"])
    @assert size(mixtures, 1) == 3 || size(mixtures, 2) == 3 # need 3-strategy mixtures
    if size(mixtures, 2) != 3
        mixtures = mixtures'
    end
    num_mixtures = size(mixtures, 1)
    @assert size(values, 1) == num_mixtures || size(values, 2) == num_mixtures
    if size(values, 1) != num_mixtures
        values = values'
    end
    to_plot = Matrix{Float64}(undef,num_mixtures,4)
    to_plot[:,1] .= mixtures[:,3]
    to_plot[:,2] .= mixtures[:,1]
    to_plot[:,3] .= mixtures[:,2]
    to_plot[:,4] .= values
    ternary(to_plot; region=(0,1,0,1,0,1), frame=(grid=0, ticks=0, annot=0), image=true)
    simplex_setup!(title; border=true, show=show, vertex_labels=vertex_labels)
end

function simplex_scatter(mixtures::AbstractVecOrMat; title=nothing, vertex_labels=["S1","S2","S3"], marker=:p, markersize=.2, kwargs...)
    simplex_setup(title; border=false, show=false, vertex_labels=vertex_labels)
    simplex_scatter!(mixtures; marker=marker, markersize=markersize, kwargs...)
end

function simplex_scatter!(mixtures::AbstractVecOrMat; marker=:p, markersize=.2, kwargs...)
    @assert size(mixtures, 1) == 3 || size(mixtures, 2) == 3 # need 3-strategy mixtures
    if size(mixtures, 2) != 3
        mixtures = mixtures'
    end
    num_mixtures = size(mixtures, 1)
    to_plot = Matrix{Float64}(undef,num_mixtures,3)
    to_plot[:,1] .= mixtures[:,3]
    to_plot[:,2] .= mixtures[:,1]
    to_plot[:,3] .= mixtures[:,2]
    plot!(tern2cart(to_plot); noclip=true, marker=marker, markersize=markersize, kwargs...)
end

function simplex_path(mixtures::AbstractVecOrMat; title=nothing, vertex_labels=["S1","S2","S3"], color=:black, width=1, kwargs...)
    simplex_setup(title; border=false, show=false, vertex_labels=vertex_labels)
    simplex_path!(mixtures; color=color, width=width, kwargs...)
end

function simplex_path!(mixtures::AbstractVecOrMat; color=:black, width=1, kwargs...)
    @assert size(mixtures, 1) == 3 || size(mixtures, 2) == 3 # need 3-strategy mixtures
    if size(mixtures, 2) != 3
        mixtures = mixtures'
    end
    num_mixtures = size(mixtures, 1)
    to_plot = Matrix{Float64}(undef,num_mixtures,3)
    to_plot[:,1] .= mixtures[:,3]
    to_plot[:,2] .= mixtures[:,1]
    to_plot[:,3] .= mixtures[:,2]
    pen=string(width)*","*string(color)
    plot!(tern2cart(to_plot); S="~n1:+sc0", par=(MAP_DEFAULT_PEN=pen,), kwargs...)
end