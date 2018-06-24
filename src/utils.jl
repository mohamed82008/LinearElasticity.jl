YoungsModulus(p::RectilinearPointLoad) = p.E
YoungsModulus(inp::InpLinearElasticity) = inp.inp_content.E
PoissonRatio(p::RectilinearPointLoad) = p.ν
PoissonRatio(inp::InpLinearElasticity) = inp.inp_content.ν
