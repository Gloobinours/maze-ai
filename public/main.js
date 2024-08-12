let maze_size = 9;
// Function to fetch maze data from the server
function get_maze(size) {
    return fetch(`http://127.0.0.1:5000/maze/${size}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json(); // Return the parsed JSON data
        })
        .then(data => {
            console.log('Fetched maze data:', data); // Log fetched data for debugging
            if (!data || !data.matrix || data.matrix.length === 0) {
                throw new Error('Fetched maze data is empty or invalid');
            }
            return data.matrix; // Return the maze matrix array
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
            throw error; // Re-throw the error to propagate it
        });
}

// Function to draw the maze on the canvas
function drawMaze(mazeArray) {
    // Maze cell dimensions and colors
    const cellSize = 15; // Size of each cell in pixels
    const wallColor = '#333'; // Color for walls
    const pathColor = '#fff'; // Color for paths
    const coinColor = '#aa0'
    const bombColor = '#a33'
    const playerColor = '#33a'

    // Get the canvas element and its context
    const canvas = document.getElementById('mazeCanvas');
    const ctx = canvas.getContext('2d');

    // Calculate the canvas dimensions based on the maze size
    const mazeWidth = mazeArray[0].length;
    const mazeHeight = mazeArray.length;
    canvas.width = mazeWidth * cellSize;
    canvas.height = mazeHeight * cellSize;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Iterate through the mazeArray and draw walls and paths
    for (let y = 0; y < mazeHeight; y++) {
        for (let x = 0; x < mazeWidth; x++) {
            if (mazeArray[y][x] === 1) {
                // Draw wall
                ctx.fillStyle = wallColor;
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            } else if(mazeArray[y][x] === 0){
                // Draw path
                ctx.fillStyle = pathColor;
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            } else if(mazeArray[y][x] === 2){
                // Draw path
                ctx.fillStyle = coinColor;
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            } else if(mazeArray[y][x] === 3){
                // Draw bomb
                ctx.fillStyle = bombColor;
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            } else if(mazeArray[y][x] === 4){
                // Draw player
                ctx.fillStyle = playerColor;
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }
        }
    }
}

// Usage: Fetch maze data and then draw it
get_maze(maze_size)
    .then(mazeData => {
        if (mazeData && mazeData.length > 0) {
            drawMaze(mazeData); // Call drawMaze with the fetched mazeData
        } else {
            throw new Error('Fetched maze data is empty or invalid');
        }
    })
    .catch(error => {
        // Handle errors if any
        console.error('Failed to fetch or draw maze:', error);
    });
