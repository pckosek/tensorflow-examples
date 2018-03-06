for /l %%x in (15, 1, 30) do (
   del model.c* che*
   python 04_simple_smoosh.py train %%x
   python 04_simple_smoosh.py test %%x
)