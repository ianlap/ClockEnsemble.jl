using Documenter
using DocumenterCitations
using Literate
using ClockEnsemble
# Plot backend: PGFPlotsX renders LaTeX-quality vector PDFs and
# font-matches the docs body. Loaded here so that any `@example` block
# that subsequently `using Plots` picks up PGFPlotsX as the default
# backend automatically (Plots.jl module state is process-global).
# Requires `pdflatex` / `lualatex` and `pdftocairo` in PATH — see
# .github/workflows/Documentation.yml for the CI install of texlive
# packages and poppler-utils.
using Plots
using PGFPlotsX
Plots.pgfplotsx()
# Enable `\text{…}` (and friends) inside math labels — PGFPlotsX's
# default preamble ships pgfplots only, not amsmath.
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsmath}")

# ── Literate.jl: render examples/*.jl into docs/src/tutorials/ ──────────────
# Each top-level `examples/*.jl` is single-source: edit the script,
# rebuild, and the matching `tutorials/<name>.md` regenerates. The
# generated pages are gitignored (only the `.jl` files are tracked).
const EXAMPLES_DIR  = joinpath(@__DIR__, "..", "examples")
const TUTORIALS_DIR = joinpath(@__DIR__, "src", "tutorials")
mkpath(TUTORIALS_DIR)
for jl in sort(readdir(EXAMPLES_DIR; join=true))
    endswith(jl, ".jl") || continue
    Literate.markdown(jl, TUTORIALS_DIR;
                      documenter = true,
                      credit     = false)
end

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style = :authoryear,
)

DocMeta.setdocmeta!(ClockEnsemble, :DocTestSetup, :(using ClockEnsemble); recursive=true)

makedocs(
    sitename = "ClockEnsemble.jl",
    modules  = [ClockEnsemble],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://ianlap.github.io/ClockEnsemble.jl",
        mathengine = Documenter.MathJax3(),
    ),
    plugins = [bib],
    pages = [
        "Home"            => "index.md",
        "Theory"          => [
            "theory/kalman.md",
            "theory/steering.md",
            "theory/ensembles.md",
        ],
        "Tutorials"       => [
            "tutorials/01_kalman_single_clock.md",
            "tutorials/02_kalman_pid_steering.md",
            "tutorials/03_holdover_comparison.md",
            "tutorials/04_process_noise_tuning.md",
        ],
        "API Reference"   => "reference.md",
        "Bibliography"    => "bibliography.md",
    ],
    doctest  = true,
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(
    repo         = "github.com/ianlap/ClockEnsemble.jl.git",
    push_preview = true,
    devbranch    = "main",
)
