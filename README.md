# Sudoku solver

## Installation
The repository can either be cloned or downloaded as a zip.

Install tesseract by following the instructions [here](https://github.com/tesseract-ocr/tessdoc/blob/main/Home.md#macos)

Python requirements can be installed using [rye](https://rye.astral.sh/guide/installation/) by running `rye sync`

## Usage
Execute the code as follows :
```python3
rye run main.py 'input.png'
```

Here the path of the input image is passed as a command line argument.

## Working


### [Image Processing using OpenCV](https://github.com/SurajSubramanian/SudokuSolver/blob/main/sudoku_solver.py)

Preprocessing sudoku image (converting to grayscale, blurring, finding contours and dividing image into 81 squares). Our [stackoverflow question](https://stackoverflow.com/q/63902994/14241803) summarises most of our work in this module.

<figure>
    <figcaption>The input sudoku image</figcaption>
    <img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/input.png" width="300" alt='missing'/>
</figure>

<figure>
    <figcaption>Converting image to grayscale</figcaption>
    <img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/gray.png" width="300" alt='missing'/>
</figure>

<figure>
    <figcaption>Finding the 4 contours of the image</figcaption>
    <img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/out.png" width="300" alt='missing'/>
</figure>

<figure>
    <figcaption>Cropping and warping the image</figcaption>
    <img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/gray2.png" width="300" alt='missing'/>
</figure>

<figure>
    <figcaption>The final image that is divided into 81 images</figcaption>
    <img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/board.png" width="300" alt='missing'/>
</figure>

<figure>
    <figcaption>Processing of a single box in the sudoku board</figcaption>
    <img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/board_01_04_initial.png" width="200" alt='missing'/>
</figure>

<figure>
    <img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/board_01_04_final.png" width="200" alt='missing'/>
</figure>

### [Digit Recognizer using pytesseract](https://github.com/SurajSubramanian/SudokuSolver/blob/main/digit_recognizer.py)

(Recognizes the digits from each of the 81 images and stores it in a list)

- The 81 images are stored in the directory [temp](https://github.com/SurajSubramanian/SudokuSolver/tree/main/temp)
- Models trained on MNIST performed poor here as these digits were not handwritten. Hence we used pytesseract. [This](https://stackoverflow.com/a/58032585/10077354) answer was helpful.
- If the box doesn't contain any number, the list holds `.` in it's place and `0-9` otherwise.

### Verification of predicted digits with user

Each digit predicted is checked with the user to see if it's predicted correctly. If not, the user will be able to correct it accordingly.

### [Backtracking](https://github.com/SurajSubramanian/SudokuSolver/blob/main/backtracking.py)

The list of length 81 is converted into a 9x9 array. Later we solve this board using backtracking and display the solved board to the user.

## OUTPUT

<img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/output1.png" alt='missing'/>

...

<img src="https://github.com/SurajSubramanian/SudokuSolver/blob/main/docs/output2.png" alt='missing'/>

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Better techniques for extracting digits from the sudoku image and digit recognition could be done.

Please make sure to update tests as appropriate.

## REFERENCES

[Python solution to solve sudoku using backtracking in leetcode](https://leetcode.com/problems/sudoku-solver/discuss/15959/Accepted-Python-solution)

[Digit Recognition using OpenCV and pytesseract](https://stackoverflow.com/a/58032585/10077354)

[Sudoku Grabber in OpenCV](https://aishack.in/tutorials/sudoku-grabber-opencv-plot/)

