#!/usr/bin/env python3
"""
run_instance_selection.py
==========================

Roda a etapa de Instance Selection (biblioteca waashk/instanceselection) sobre
os datasets que ja estao preparados dentro de RAG-Fuse/resource/dataset/, para
QUALQUER um dos metodos suportados pela lib, passado por parametro.

Isso replica (e generaliza) o que a celula "Instance Selection" do notebook
rag_fuse.ipynb ja fazia manualmente so para o LSSm:

  1. Le o split original (10 ou 5 folds) de
     resource/dataset/raw_dataset/<dataset>/splits/split_<nfolds>_with_val.pkl
  2. Le resource/dataset/<dataset>/samples.pkl (o corpus completo, com texto e label)
  3. Para cada fold, vetoriza o TREINO com TF-IDF e roda o metodo de IS,
     obtendo os indices que sobrevivem a selecao
  4. Traduz esses indices de volta para os IDs globais do RAG-Fuse
  5. Grava um NOVO dataset (resource/dataset/<dataset>_is_<method>/) com a
     mesma estrutura fold_0..fold_N/{train,val,test,labels_descriptions}.pkl
     que o RAG-Fuse espera, so o train.pkl muda de tamanho, val/test e
     labels_descriptions sao copiados sem alteracao.
  6. (opcional) Gera um novo setting/data/<dataset>_is_<method>.yaml apontando
     "dir" para essa nova pasta, para voce rodar `data=<dataset>_is_<method>`
     no main.py / nos scripts run/*.sh sem tocar no dataset original.

Por que um dataset NOVO em vez de sobrescrever o original?
------------------------------------------------------------
Observacao importante: depois do Instance Selection o tamanho do
treino muda (teste e validacao NAO mudam, IS so reduz o treino). Se voce
sobrescrever resource/dataset/<dataset>/fold_X/train.pkl direto, perde o
dataset original e o `dir:` configurado em setting/data/<dataset>.yaml passa
a apontar silenciosamente para dados diferentes dos que voce usou na etapa de
otimizacao de prompt. Por isso este script, por padrao, cria uma pasta nova
(<dataset>_is_<method>) e, se voce pedir --update-config, cria tambem um yaml
novo, ou seja, o "local de onde pegar o train.pkl" (o campo `dir` do yaml de
config) e o que voce troca para apontar para os dados pos-selecao, mantendo o
dataset original intocado. Use --inplace apenas se tiver certeza.

Metodos suportados (--method): cnn, enn, icf, lssm, lsbo, drop3, ldis, cdis,
xldis, psdsp, ib3, cis, egdis

Exemplos
--------
# Rodar LSSm nos datasets acm, ohsumed e reut90, criando resource/dataset/<d>_is_lssm/
python run_instance_selection.py \\
    --ragfuse-dir /content/RAG-Fuse \\
    --iselib-dir /content/instanceselection \\
    --method lssm \\
    --datasets acm ohsumed reut90 \\
    --update-config

# Rodar em TODOS os datasets que tem raw_dataset/splits disponivel
python run_instance_selection.py --ragfuse-dir . --iselib-dir ../instanceselection --method egdis --update-config

# LSBo depende do LSSm internamente (a propria lib exige isso), o script cuida
# disso sozinho, NAO precisa rodar lssm antes a parte:
python run_instance_selection.py --ragfuse-dir . --iselib-dir ../instanceselection --method lsbo
"""

import argparse
import copy
import pickle
import shutil
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_METHODS = [
    "cnn", "enn", "icf", "lssm", "lsbo", "drop3",
    "ldis", "cdis", "xldis", "psdsp", "ib3", "cis", "egdis",
]

# Datasets conhecidamente sem raw_dataset/splits (sem esse pre-requisito NAO da
# para rodar IS neles, precisam ser pulados). No repo atual isso vale para o
# "twitter" (comentado no notebook original como "ERRO - NAO possuo sem labels").
KNOWN_UNSUPPORTED = {"twitter"}


# ---------------------------------------------------------------------------
# Compatibilidade com sklearn/six modernos (a lib instanceselection foi escrita
# para sklearn antigo e python 3.6). Em vez de editar os arquivos do repo
# clonado (como o notebook fazia com !sed -i), fazemos isso via monkeypatch em
# tempo de execucao, o que funciona NAO importa qual metodo for escolhido.
# ---------------------------------------------------------------------------
def patch_sklearn_compat():
    import sklearn.neighbors as skl_neighbors

    if not hasattr(skl_neighbors, "classification"):
        mod = types.ModuleType("sklearn.neighbors.classification")
        mod.KNeighborsClassifier = skl_neighbors.KNeighborsClassifier
        sys.modules["sklearn.neighbors.classification"] = mod
        skl_neighbors.classification = mod

    try:
        import sklearn.externals as skl_externals
    except ImportError:
        skl_externals = types.ModuleType("sklearn.externals")
        sys.modules["sklearn.externals"] = skl_externals

    if not hasattr(skl_externals, "six"):
        try:
            import six
        except ImportError:
            raise SystemExit(
                "O pacote 'six' NAO esta instalado. Rode: pip install six"
            )
        sys.modules["sklearn.externals.six"] = six
        skl_externals.six = six


def import_iselib(iselib_dir: Path):
    """Adiciona a lib instanceselection ao sys.path e importa os seletores."""
    iselib_dir = str(iselib_dir.resolve())
    if iselib_dir not in sys.path:
        sys.path.insert(0, iselib_dir)

    patch_sklearn_compat()

    from src.main.python.iSel import (
        cnn, enn, icf, lssm, lsbo, drop3,
        ldis, cdis, xldis, psdsp, ib3, cis, egdis,
    )
    return {
        "cnn": cnn, "enn": enn, "icf": icf, "lssm": lssm, "lsbo": lsbo,
        "drop3": drop3, "ldis": ldis, "cdis": cdis, "xldis": xldis,
        "psdsp": psdsp, "ib3": ib3, "cis": cis, "egdis": egdis,
    }


class _DummyArgs:
    """Objeto minimo exigido pelo __init__ de LSBo/DROP3 (so usam .outputdir)."""
    def __init__(self, outputdir):
        self.outputdir = str(outputdir)


def get_selector(method: str, modules: dict, dummy_args: "_DummyArgs", fold: int):
    if method == "cnn":
        return modules["cnn"].CNN()
    if method == "enn":
        return modules["enn"].ENN()
    if method == "icf":
        return modules["icf"].ICF()
    if method == "lssm":
        return modules["lssm"].LSSm()
    if method == "lsbo":
        return modules["lsbo"].LSBo(dummy_args, fold)
    if method == "drop3":
        return modules["drop3"].DROP3(dummy_args, fold, n_neighbors=3, loadenn=False)
    if method == "ldis":
        return modules["ldis"].LDIS()
    if method == "cdis":
        return modules["cdis"].CDIS()
    if method == "xldis":
        return modules["xldis"].XLDIS()
    if method == "psdsp":
        return modules["psdsp"].PSDSP()
    if method == "ib3":
        return modules["ib3"].IB3()
    if method == "cis":
        return modules["cis"].CIS(task="atc")
    if method == "egdis":
        return modules["egdis"].EGDIS()
    raise ValueError(f"Metodo desconhecido: {method}")


# ---------------------------------------------------------------------------
# Helpers de I/O do RAG-Fuse
# ---------------------------------------------------------------------------
def get_splits(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_samples(dataset_dir: Path):
    with open(dataset_dir / "samples.pkl", "rb") as f:
        return pd.DataFrame(pickle.load(f))


def find_raw_split_file(raw_dataset_dir: Path, nfolds: int):
    candidates = [
        raw_dataset_dir / "splits" / f"split_{nfolds}_with_val.pkl",
        raw_dataset_dir / "splits" / f"split_{nfolds}.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def discover_datasets(resource_dataset_dir: Path, nfolds: int):
    raw_root = resource_dataset_dir / "raw_dataset"
    found = []
    if not raw_root.exists():
        return found
    for d in sorted(raw_root.iterdir()):
        if not d.is_dir() or d.name in KNOWN_UNSUPPORTED:
            continue
        if find_raw_split_file(d, nfolds) is not None:
            found.append(d.name)
    return found


# ---------------------------------------------------------------------------
# Nucleo: rodar IS para um dataset
# ---------------------------------------------------------------------------
def vectorize_fold(texts, max_features, stop_words, min_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        min_df=min_df,
    )
    X = vectorizer.fit_transform(texts)
    return X


def run_lssm_local_mask(X_train_tfidf, y_train, modules):
    """Roda LSSm e devolve os indices LOCAIS (posicao dentro do fold, sem
    traducao para IDs globais), e exatamente isso que o LSBo espera
    encontrar no arquivo split_10_lssm_idxinfold.pkl."""
    selector = modules["lssm"].LSSm()
    selector.fit(X_train_tfidf, y_train)
    return np.asarray(selector.sample_indices_)


def run_instance_selection_for_dataset(
    dataset: str,
    ragfuse_dir: Path,
    modules: dict,
    method: str,
    nfolds: int,
    max_features: int,
    stop_words,
    min_df: int,
    cache_dir: Path,
    seed: int,
):
    resource_dataset_dir = ragfuse_dir / "resource" / "dataset"
    dataset_dir = resource_dataset_dir / dataset
    raw_dataset_dir = resource_dataset_dir / "raw_dataset" / dataset

    split_file = find_raw_split_file(raw_dataset_dir, nfolds)
    if split_file is None:
        print(f"[{dataset}] AVISO: NAO achei raw_dataset/{dataset}/splits/split_{nfolds}(_with_val).pkl, pulando.")
        return None

    splits_df = get_splits(split_file)
    samples_df = load_samples(dataset_dir)
    samples_df = samples_df.set_index("idx", drop=False)

    has_val = "val_idxs" in splits_df.columns

    reduced_splits = copy.deepcopy(splits_df)
    stats = []

    dummy_args = _DummyArgs(outputdir=cache_dir)

    for row_idx, row in splits_df.iterrows():
        fold = int(row["fold_id"]) if "fold_id" in splits_df.columns else int(row_idx)
        train_idxs = list(row["train_idxs"])

        train_data = samples_df.loc[train_idxs]
        X_train_text = train_data["text"].tolist()
        y_train = train_data["labels_ids"].apply(lambda x: x[0]).to_numpy().ravel()

        X_train_tfidf = vectorize_fold(X_train_text, max_features, stop_words, min_df)

        np.random.seed(seed)

        if method == "lsbo":
            # LSBo exige um arquivo com os indices LOCAIS que o LSSm manteve
            # neste fold (a propria lib faz isso, ver src/main/python/iSel/lsbo.py).
            # Calculamos aqui e escrevemos no cache com o nome fixo que o
            # codigo da lib espera.
            local_mask = run_lssm_local_mask(X_train_tfidf, y_train, modules)
            cache_path = Path(dummy_args.outputdir)
            cache_path.mkdir(parents=True, exist_ok=True)
            # A lib recarrega esse arquivo inteiro a cada fold via
            # get_splits(...).loc[self.fold], entao mantemos/atualizamos
            # incrementalmente as linhas ja computadas (uma por fold).
            _merge_lssm_cache(cache_path / "split_10_lssm_idxinfold.pkl", fold, local_mask)

        selector = get_selector(method, modules, dummy_args, fold)

        t_original = len(y_train)
        selector.fit(X_train_tfidf, y_train)
        local_selected = np.asarray(selector.sample_indices_)

        new_train_idxs = [train_idxs[i] for i in local_selected]

        reduced_splits.at[row_idx, "train_idxs"] = new_train_idxs

        t_reduced = len(new_train_idxs)
        reduction = (t_original - t_reduced) / t_original if t_original else 0.0
        stats.append({
            "dataset": dataset, "fold": fold,
            "original_size": t_original, "reduced_size": t_reduced,
            "reduction_pct": round(reduction * 100, 2),
        })
        print(f"[{dataset}] fold {fold}: {t_original} -> {t_reduced} "
              f"instancias de treino (reducao {reduction*100:.1f}%)")

    return {
        "reduced_splits": reduced_splits,
        "has_val": has_val,
        "stats": stats,
    }


def _merge_lssm_cache(cache_file: Path, fold: int, local_mask: np.ndarray):
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
        df.loc[fold] = {"train_idxs": local_mask}
    else:
        df = pd.DataFrame({"train_idxs": [local_mask]}, index=[fold])
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)


# ---------------------------------------------------------------------------
# Materializar o novo dataset em resource/dataset/<dataset><suffix>/
# ---------------------------------------------------------------------------
def materialize_dataset(
    dataset: str,
    ragfuse_dir: Path,
    reduced_splits: pd.DataFrame,
    out_suffix: str,
    inplace: bool,
):
    resource_dataset_dir = ragfuse_dir / "resource" / "dataset"
    src_dataset_dir = resource_dataset_dir / dataset

    if inplace:
        dst_dataset_dir = src_dataset_dir
    else:
        dst_dataset_dir = resource_dataset_dir / f"{dataset}{out_suffix}"
        dst_dataset_dir.mkdir(parents=True, exist_ok=True)
        # Arquivos do corpus inteiro, compartilhados entre todos os folds,
        # NAO mudam com instance selection: copiamos como estao.
        for fname in ("samples.pkl", "relevance_map.pkl", "label_cls.pkl", "text_cls.pkl"):
            src_f = src_dataset_dir / fname
            if src_f.exists():
                shutil.copy2(src_f, dst_dataset_dir / fname)

    for row_idx, row in reduced_splits.iterrows():
        fold = int(row["fold_id"]) if "fold_id" in reduced_splits.columns else int(row_idx)
        fold_dir = dst_dataset_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # train.pkl: o unico que muda de fato
        with open(fold_dir / "train.pkl", "wb") as f:
            pickle.dump(list(row["train_idxs"]), f)

        if not inplace:
            src_fold_dir = src_dataset_dir / f"fold_{fold}"
            # val/test/labels_descriptions: copiados sem alteracao
            for fname in ("val.pkl", "test.pkl", "labels_descriptions.pkl"):
                src_f = src_fold_dir / fname
                if src_f.exists():
                    shutil.copy2(src_f, fold_dir / fname)

    return dst_dataset_dir


def update_hydra_config(ragfuse_dir: Path, dataset: str, new_dataset_dir_name: str):
    """Cria setting/data/<dataset><suffix>.yaml apontando `dir` para o dataset
    pos-instance-selection, preservando os demais campos do yaml original.
    e este `dir` que o "doutor" comentou que talvez precisasse mudar depois
    do Instance Selection, o RetrieverDataModule le train.pkl/val.pkl/test.pkl
    de `{data.dir}fold_X/...`, entao apontar esse campo para a pasta nova e o
    suficiente para o pipeline do RAG-Fuse (run.sh / main.py) ja usar os dados
    reduzidos sem tocar no dataset original.
    """
    setting_data_dir = ragfuse_dir / "setting" / "data"
    src_yaml = setting_data_dir / f"{dataset}.yaml"
    if not src_yaml.exists():
        print(f"[{dataset}] AVISO: NAO achei setting/data/{dataset}.yaml, NAO vou gerar config novo.")
        return None

    new_name = new_dataset_dir_name
    dst_yaml = setting_data_dir / f"{new_name}.yaml"

    text = src_yaml.read_text(encoding="utf-8")
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("name:"):
            new_lines.append(f"name: {new_name}")
        elif line.startswith("dir:"):
            new_lines.append(f"dir: resource/dataset/{new_name}/")
        else:
            new_lines.append(line)
    dst_yaml.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[{dataset}] config novo criado em {dst_yaml} "
          f"(use `data={new_name}` no run.sh / main.py)")
    return dst_yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Roda Instance Selection sobre os datasets do RAG-Fuse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ragfuse-dir", type=Path, required=True,
                    help="Raiz do repositorio RAG-Fuse (contem resource/, setting/, main.py).")
    p.add_argument("--iselib-dir", type=Path, required=True,
                    help="Raiz do repositorio instanceselection (git clone waashk/instanceselection).")
    p.add_argument("-m", "--method", required=True, choices=SUPPORTED_METHODS,
                    help="Metodo de instance selection a aplicar.")
    p.add_argument("-d", "--datasets", nargs="+", default=None,
                    help="Datasets a processar (nomes das pastas em resource/dataset/). "
                         "Se omitido, processa todos os que tiverem raw_dataset/<d>/splits disponivel.")
    p.add_argument("--nfolds", type=int, default=10, help="Numero de folds (default: 10).")
    p.add_argument("--max-features", type=int, default=10000, help="max_features do TF-IDF.")
    p.add_argument("--stop-words", default=None, choices=[None, "english"],
                    help="Remover stopwords no TF-IDF (README oficial da lib usa 'english'). Default: nenhum.")
    p.add_argument("--min-df", type=int, default=1,
                    help="min_df do TF-IDF (README oficial usa 2: mantem so termos em >=2 docs). Default: 1.")
    p.add_argument("--seed", type=int, default=1608637542,
                    help="Seed (default = mesma usada pela lib original).")
    p.add_argument("--out-suffix", default=None,
                    help="Sufixo da pasta de saida resource/dataset/<dataset><suffix>/. "
                         "Default: '_is_<method>'.")
    p.add_argument("--inplace", action="store_true",
                    help="PERIGO: sobrescreve resource/dataset/<dataset>/fold_*/train.pkl original "
                         "em vez de criar uma pasta nova.")
    p.add_argument("--update-config", action="store_true",
                    help="Gera setting/data/<dataset><suffix>.yaml apontando para o dataset novo.")
    p.add_argument("--cache-dir", type=Path, default=None,
                    help="Pasta de cache/staging (usada internamente, ex. pelo LSBo). "
                         "Default: <ragfuse-dir>/.instance_selection_cache")
    return p.parse_args()


def main():
    args = parse_args()

    ragfuse_dir = args.ragfuse_dir.resolve()
    iselib_dir = args.iselib_dir.resolve()
    resource_dataset_dir = ragfuse_dir / "resource" / "dataset"

    if not resource_dataset_dir.exists():
        raise SystemExit(f"NAO achei {resource_dataset_dir}, confira --ragfuse-dir.")
    if not (iselib_dir / "src" / "main" / "python" / "iSel").exists():
        raise SystemExit(
            f"NAO achei a lib instanceselection em {iselib_dir}. "
            "Clone com: git clone https://github.com/waashk/instanceselection.git"
        )

    modules = import_iselib(iselib_dir)

    datasets = args.datasets or discover_datasets(resource_dataset_dir, args.nfolds)
    if not datasets:
        raise SystemExit("Nenhum dataset encontrado/informado para processar.")

    out_suffix = args.out_suffix or f"_is_{args.method}"
    cache_dir = args.cache_dir or (ragfuse_dir / ".instance_selection_cache")

    print(f"Metodo: {args.method}")
    print(f"Datasets: {datasets}")
    print(f"Saida: {'IN-PLACE (sobrescrevendo original!)' if args.inplace else 'resource/dataset/<d>' + out_suffix}")
    print()

    all_stats = []
    for dataset in datasets:
        cache_dir_ds = cache_dir / dataset
        result = run_instance_selection_for_dataset(
            dataset=dataset,
            ragfuse_dir=ragfuse_dir,
            modules=modules,
            method=args.method,
            nfolds=args.nfolds,
            max_features=args.max_features,
            stop_words=args.stop_words,
            min_df=args.min_df,
            cache_dir=cache_dir_ds,
            seed=args.seed,
        )
        if result is None:
            continue

        dst_dir = materialize_dataset(
            dataset=dataset,
            ragfuse_dir=ragfuse_dir,
            reduced_splits=result["reduced_splits"],
            out_suffix=out_suffix,
            inplace=args.inplace,
        )
        print(f"[{dataset}] dataset pos-IS salvo em: {dst_dir}\n")

        if args.update_config and not args.inplace:
            update_hydra_config(ragfuse_dir, dataset, dst_dir.name)

        all_stats.extend(result["stats"])

    if all_stats:
        summary = pd.DataFrame(all_stats)
        by_dataset = summary.groupby("dataset").agg(
            original_size=("original_size", "sum"),
            reduced_size=("reduced_size", "sum"),
        )
        by_dataset["kept_pct"] = (by_dataset["reduced_size"] / by_dataset["original_size"] * 100).round(2)
        by_dataset["reduction_pct"] = (100 - by_dataset["kept_pct"]).round(2)
        print("\n=== Resumo (todos os folds somados) ===")
        print(by_dataset.to_string())

        summary_path = cache_dir / f"summary_{args.method}.csv"
        cache_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"\nDetalhe por fold salvo em: {summary_path}")


if __name__ == "__main__":
    main()
