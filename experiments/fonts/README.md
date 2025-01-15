If you are running the experiments on Ubuntu, the font may not be recognized by matplotlib's fontManager. To resolve this, place the font file in the directory containing matplotlib's fonts and restart your virtual environment.

The directory can be found within the virtual environment path. For example:
`fedmoe_env/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf`.

If this does not resolve the issue, remove matplotlib's cache (`~/.cache/matplotlib`), and then rerun the visualizations.