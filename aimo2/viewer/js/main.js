async function loadAndRenderMath() {
    // Load and parse records
    const fileSelect = document.querySelector('select');
    if (!fileSelect.value) {
        return;
    }
    const records = await loadJsonlFile(fileSelect.value);

    // Group and render records
    const content = document.getElementById('content');
    content.innerHTML = '';

    const questionGroups = Object.entries(groupRecordsByQuestion(records))
        .sort(([a], [b]) => a.localeCompare(b));

    for (const [id, group] of questionGroups) {
        content.appendChild(renderQuestionGroup(group));
    }
}

async function loadAndRender() {
    // Create and add file selector
    const fileSelect = await createFileSelector();
    document.body.insertBefore(fileSelect, document.getElementById('content'));

    // Update when selection changes
    fileSelect.addEventListener('change', () => {
        loadAndRenderMath();
    });

    const options = {
        throwOnError: false,
        nonStandard: true
    };

    marked.use(markedKatex(options));
}

// Start the application
document.addEventListener('DOMContentLoaded', loadAndRender); 