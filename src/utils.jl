YoungsModulus(p::RectilinearPointLoad) = p.E
YoungsModulus(inp::InpLinearElasticity) = inp.inp_content.E
