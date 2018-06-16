function assemble(problem::LinearElasticityProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T}) where {dim,T}
    globalinfo = GlobalFEAInfo(problem)
    assemble!(globalinfo, problem, elementinfo)
    return globalinfo
end

function assemble!(globalinfo::GlobalFEAInfo{T}, problem::LinearElasticityProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T, TK}) where {dim, T, TK}
    ch = problem.ch
    dh = ch.dh
    K, f = globalinfo.K, globalinfo.f
    f .= elementinfo.fixedload
    Kes, fes = elementinfo.Kes, elementinfo.fes
    if K isa Symmetric
        K.data.nzval .= 0
        assembler = JuAFEM.AssemblerSparsityPattern(K.data, f, Int[], Int[])
    else
        K.nzval .= 0
        assembler = JuAFEM.AssemblerSparsityPattern(K, f, Int[], Int[])
    end
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    fe = zeros(fes[1])
    Ke = zeros(T, size(Kes[1]))
    celliteratortype = CellIterator{typeof(dh).parameters...}
    _celliterator::celliteratortype = CellIterator(dh)
    for (i,cell) in enumerate(_celliterator)
        celldofs!(global_dofs, dh, i)
        if TK <: Symmetric
            JuAFEM.assemble!(assembler, global_dofs, Kes[i].data, fes[i])
        else
            JuAFEM.assemble!(assembler, global_dofs, Kes[i], fes[i])
        end
    end
    if TK <: Symmetric
        apply!(K.data, f, ch)
    else
        apply!(K, f, ch)
    end

    return 
end

function assemble_f(problem::LinearElasticityProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T}) where {dim, T}
    f = zeros(T, ndofs(problem.ch.dh))
    assemble_f!(f, problem, elementinfo)
    return f
end

function assemble_f!(f::AbstractVector, problem::LinearElasticityProblem, elementinfo::ElementFEAInfo)
    fes = elementinfo.fes
    f .= elementinfo.fixedload
    dof_cells = problem.metadata.dof_cells
    dof_cells_offset = problem.metadata.dof_cells_offset
    for dofidx in 1:ndofs(problem.ch.dh)
        r = dof_cells_offset[dofidx] : dof_cells_offset[dofidx+1]-1
        for i in r
            cellidx, localidx = dof_cells[i]
            f[dofidx] += fes[cellidx][localidx]
        end
    end
    return f
end

function assemble_f!(f::AbstractVector, problem, dloads)
    metadata = problem.metadata
    dof_cells = metadata.dof_cells
    dof_cells_offset = metadata.dof_cells_offset
    for dofidx in 1:ndofs(problem.ch.dh)
        r = dof_cells_offset[dofidx] : dof_cells_offset[dofidx+1]-1
        for i in r
            cellidx, localidx = dof_cells[i]
            f[dofidx] += dloads[cellidx][localidx]
        end
    end
    return f
end
