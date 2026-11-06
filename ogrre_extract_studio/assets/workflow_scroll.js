// assets/workflow_scroll.js

(function () {
    function updateWorkflowMasks() {
        const containers = document.querySelectorAll('.workflow-container');
        containers.forEach(el => {
            // Distance (in pixels) from the absolute bottom of the scroll container
            const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;

            // If within 15px of the bottom (or if content fits without scrolling)
            if (distanceFromBottom <= 15) {
                if (!el.classList.contains('at-bottom')) {
                    el.classList.add('at-bottom');
                }
            } else {
                if (el.classList.contains('at-bottom')) {
                    el.classList.remove('at-bottom');
                }
            }
        });
    }

    // 1. Capture scroll events across the document
    document.addEventListener('scroll', function (e) {
        if (e.target && e.target.classList && e.target.classList.contains('workflow-container')) {
            updateWorkflowMasks();
        }
    }, true);

    // 2. Observe DOM mutations (catches dynamic Dash callback updates)
    const observer = new MutationObserver(updateWorkflowMasks);
    document.addEventListener('DOMContentLoaded', () => {
        observer.observe(document.body, { childList: true, subtree: true, attributes: true });
        updateWorkflowMasks();
    });

    // 3. Interval check to guarantee sync during scrollbar drags and window resizes
    setInterval(updateWorkflowMasks, 150);
    window.addEventListener('resize', updateWorkflowMasks);
})();