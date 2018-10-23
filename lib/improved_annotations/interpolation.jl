using Pkg
using DataFrames
using DataFramesMeta
using Queryverse
using CSV
using DelimitedFiles

function interpolate(dataset_slice, destination)
    # takes in a Dataframe slice and 
    println(dataset_slice)
end

for i in 1:20
    padded_index = lpad(string(i), 2, string(0))
    anns = readtable("/home/drussel1/data/ADL/ADL_annotations/object_annotation/object_annot_P_$(padded_index).txt", header = false, separator = ' ', names = [ Symbol("ID"), Symbol("xmin"), Symbol("ymin"), Symbol("xmax"), Symbol("ymax"), Symbol("frame"), Symbol("is_active"), Symbol("class")])

    sort!(anns, :ID)
    unique_IDs = unique(anns[:,:ID]) 
    for i in unique_IDs
        println(i)
        slice_ = anns[anns[:ID] .== i, :]
        interpolate(slice_)
    end
    
    output = hcat(anns[:frame], anns[:ID], anns[:xmin], anns[:ymin], anns[:xmax] - anns[:xmin], anns[:ymax] - anns[:ymin], ones(size(anns)[1],1), zeros(size(anns)[1],3)) # , zeros(size(anns)[1],3))
    writedlm("output$(padded_index).csv", output, ' ')
end
