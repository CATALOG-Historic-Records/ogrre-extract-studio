// assets/zoom_handler.js
window.addEventListener('wheel', function (e) {
    const pdfGraph = document.getElementById('pdf-graph');
    if (!pdfGraph || !pdfGraph.contains(e.target)) return;

    // Trigger zoom only when Ctrl key is held down
    if (e.ctrlKey) {
        e.preventDefault();

        const plotlyPlot = pdfGraph.querySelector('.js-plotly-plot');
        if (!plotlyPlot || !plotlyPlot.layout) return;

        const factor = e.deltaY > 0 ? 1.15 : 0.85; // Zoom scale step
        const xRange = plotlyPlot.layout.xaxis.range;
        const yRange = plotlyPlot.layout.yaxis.range;

        if (xRange && yRange) {
            const xCenter = (xRange[0] + xRange[1]) / 2;
            const yCenter = (yRange[0] + yRange[1]) / 2;
            const xHalf = ((xRange[1] - xRange[0]) / 2) * factor;
            const yHalf = ((yRange[1] - yRange[0]) / 2) * factor;

            Plotly.relayout(plotlyPlot, {
                'xaxis.range': [xCenter - xHalf, xCenter + xHalf],
                'yaxis.range': [yCenter - yHalf, yCenter + yHalf]
            });
        }
    }
}, { passive: false });