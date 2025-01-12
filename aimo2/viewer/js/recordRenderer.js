function groupRecordsByQuestion(records) {
    const questionGroups = {};
    records.forEach(record => {
        if (!questionGroups[record.id]) {
            questionGroups[record.id] = [];
        }
        questionGroups[record.id].push(record);
    });
    return questionGroups;
}

function renderQuestionGroup(group) {
    const questionDiv = document.createElement('div');
    questionDiv.style.marginBottom = '2em';
    questionDiv.style.borderBottom = '1px solid #ccc';
    questionDiv.style.padding = '1em';

    // Show id
    const question = document.createElement('div');
    question.style.fontWeight = 'bold';
    question.textContent = group[0].id;
    questionDiv.appendChild(question);

    // Show each system's response
    group.forEach(record => {
        questionDiv.appendChild(renderResponse(record));
    });

    return questionDiv;
}

function renderResponse(record) {
    const responseDiv = document.createElement('details');
    responseDiv.style.marginTop = '1em';
    responseDiv.style.marginLeft = '2em';

    // Create summary (header) element with system name and answers
    const summary = document.createElement('summary');
    summary.style.cursor = 'pointer';
    summary.style.display = 'flex';
    summary.style.gap = '1em';
    summary.style.alignItems = 'center';

    const systemName = document.createElement('strong');
    systemName.textContent = `System ${record.system_name}`;
    summary.appendChild(systemName);

    const length = document.createElement('span');
    length.textContent = `Length: ${record.output_text.length}`;
    summary.appendChild(length);

    const answers = document.createElement('span');
    answers.textContent = `Answers: ${record.answers.join(', ')}`;
    summary.appendChild(answers);

    if (record.correct_answer !== null) {
        const correct = document.createElement('span');
        correct.textContent = `Correct: ${record.correct_answer}`;
        summary.appendChild(correct);
    }

    responseDiv.appendChild(summary);

    // Content container (only shown when expanded)
    const content = document.createElement('div');
    content.style.marginTop = '0.5em';
    content.style.marginLeft = '1em';

    const inputText = document.createElement('pre');
    inputText.style.fontFamily = 'monospace';
    inputText.style.whiteSpace = 'pre-wrap';
    inputText.textContent = record.input_text;
    content.appendChild(inputText);

    const response = document.createElement('div');
    text = record.output_text;
    text = text.replace(/\\\(/g, '$');
    text = text.replace(/\\\)/g, '$');
    text = text.replace(/\\\[/g, '$$');
    text = text.replace(/\\\]/g, '$$');
    response.innerHTML = marked.parse(text);
    content.appendChild(response);

    responseDiv.appendChild(content);
    return responseDiv;
} 