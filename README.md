
# Hankel

A command-line tool for learning non-deterministic Weighted Finite State Automata with Spectral Learning.

<table align="center" style="border-collapse: collapse; border: 0px solid transparent !important;">
<tr style="border: 0px solid transparent !important;">
<td valign="middle" style="border: 0px solid transparent !important; padding: 10px;"><img src="./wfsa.png" alt="tomita-5-WFSA" width="100%"/></td>
<td valign="middle" style="border: 0px solid transparent !important; padding: 10px;"><img src="./fsa.png" alt="tomita-5-FSA" width="100%"/></td>
</tr>
</table>

## Installation

```bash
sudo apt install graphviz
```

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install git+https://github.com/JoseLlarena/hankel.git
```

This will install the executables `hankel-learn`, `hankel-predict` and `hankel-show`.

## Usage

```bash
git clone git@github.com:JoseLlarena/hankel.git
cd hankel
```

### Acceptor

```bash
hankel-learn ./examples/data/tomita_2_pos_neg_lab.txt ./tomita_2.npz
hankel-show -o cons,./wfsa.png ./tomita_2.npz
hankel-predict -s -o ./prediction-5-by-2.txt ./examples/data/tomita_5_pos_unlab.txt ./tomita_2.npz
```

### Language model

```bash
hankel-learn -k lm ./examples/data/tomita_2_pos_unlab.txt ./tomita_2_lm.npz
hankel-show -o cons,./lm.png ./tomita_2_lm.npz
hankel-predict -s -o ./prediction-5-by-2-lm.txt ./examples/data/tomita_5_pos_unlab.txt ./tomita_2_lm.npz
```

For full usage documentation call commands with `--help`.

## Features

- Learning of WFSAs and FSAs
- Learning of acceptors and language models
- Export of models to console, png and pdf
- Choice of basis selection algorithms: frequency-based, length-based, pmi-based and all-affixes
- Example scripts and data for the Reber grammar, all Tomita grammars and all recursive binary boolean languages

Currently only capable of handling small vocabularies.

## Testing

```bash
python3.10 -m venv venv
source venv/bin/activate
git clone git@github.com:JoseLlarena/hankel.git
cd hankel
pip install -r requirements.txt -r requirements-test.txt
pytest --cov=hankel unittests/*.py --cov-report=term --cov-report=html && python -m webbrowser htmlcov/index.html 
```

## References

```
Balle, B., Carreras, X., Luque, F. M., & Quattoni, A. (2014). 
Spectral learning of weighted automata: A forward-backward perspective. 
Machine Learning, 96(1–2), 33–63. https://doi.org/10.1007/s10994-013-5416-x
```

```
Learning Weighted Automata. (2015). 
In B. Balle & M. Mohri, Lecture Notes in Computer Science (pp. 1–21). 
Springer International Publishing. https://doi.org/10.1007/978-3-319-23021-4_1
```

```
Rabusseau, G., Li, T., & Precup, D. (2019). 
Connecting Weighted Automata and Recurrent Neural Networks through Spectral Learning. 
Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, 1630–1639. 
https://proceedings.mlr.press/v89/rabusseau19a.html
```

## Changelog

Check the [Changelog](https://github.com/JoseLlarena/hankel/blob/main/CHANGELOG.md) for fixes and enhancements of each version.

## License

Copyright Jose Llarena 2025.

Distributed under the terms of the [MIT](https://github.com/JoseLlarena/hankel/blob/main/LICENSE) license, Hankel is free
and open source software.
