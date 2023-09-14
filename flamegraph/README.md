# Flamegraph filter and analysis tool

This tool reads 'collapsed' flamegraphs in the format produced by Brendan Gregg's `stackcollapse.pl` tools.
This lets us consume flamegraphs produced by a variety of profiling tools, filter/annotate them as we see fit, 
and then feed them to Brendan Gregg's `flamegraph.pl` SVG visualizer. 

To use:
```bash
phasm-flamegraph perf.folded > perf.filtered
```


