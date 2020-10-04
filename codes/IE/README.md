## Usage
Set up two OIE systems 
- https://github.com/philipperemy/Stanford-OpenIE-Python
- https://github.com/mmxgn/miniepy

And then put/replace main.py to the suitable position and run.
- Stanford OpenIE path: `Stanford-OpenIE-Python/`
- MinIE path: `miniepy/src/main/python/tests/minie/`
- `python main.py`

Parameters:
- `extract`: the corpus file used for information extraction
- `concat`: concatenating sentences within a story
- `random`: random sentences within a story
- `guess`: guess names from the story text
- `te`: test mode
- `te_ind`: test index of story

e.g. `python main.py` 
`--extract corpus/6.2_train.csv`
`--concat`
`--random`
`--guess`
`--te --te_ind 1 2 3`
