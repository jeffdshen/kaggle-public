async function createFileSelector() {
    const fileSelect = document.createElement('select');
    fileSelect.style.marginBottom = '1em';

    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.text = '-- Select a file --';
    fileSelect.appendChild(defaultOption);

    // Fetch list of jsonl files
    const logsResponse = await fetch('logs/');
    const logFiles = await logsResponse.text();
    const jsonlFiles = logFiles.match(/href="([\w-]+\.jsonl)"/g)?.map(match => match.slice(6, -1)) || [];

    // Add options for each jsonl file
    jsonlFiles.forEach(file => {
        const option = document.createElement('option');
        option.value = `logs/${file}`;
        option.text = file;
        fileSelect.appendChild(option);
    });

    return fileSelect;
}

async function loadJsonlFile(filepath) {
    const response = await fetch(filepath);
    const text = await response.text();
    return text.trim().split('\n')
        .map(line => JSON.parse(line));
} 