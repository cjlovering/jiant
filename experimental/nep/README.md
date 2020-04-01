From the `nep` folder.

Use `raw.py` to generate the raw `.tsv` files in `/raw/`.
Use `nli.py` to generate the `nli` pairs  in `/nli/`.

```
cd NegNN                        # 1
python -m pip install -e .      # 2
cd ..                           # 3
python NEP/raw.py               # 4
```