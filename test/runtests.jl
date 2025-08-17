using SimpleAL
using AmplNLReader
using Test

@testset "toyprob mpec" begin
    nl = AmplModel("toyprob.nl")
    output = al(nl, verbose=0)
    @test output.status == 0
end

@testset "toyprob nlp" begin
    nl = AmplModel("toyprob.nl")
    nlp = AmplMPECModel(nl)
    output = al(nlp, verbose=0)
    @test output.status == 2
end
