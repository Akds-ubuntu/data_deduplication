import logging
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table


from dedup_tool.config import DedupConfig
from dedup_tool.core.strategy import DedupStrategy
from dedup_tool.core.strategyregistry import StrategyRegistry


console = Console()
logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(console=console, show_path=False)]
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Deduplication tool using MinHash and LSH",
    no_args_is_help=True,
)


def get_strategy(config: DedupConfig) -> DedupStrategy:
    """Factory function to get the right deduplication strategy."""
    try:
        strategy_class = StrategyRegistry.get(config.algorithm)
    except KeyError:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    return strategy_class.from_config(config)

@app.command()
def test(
    config_file: Optional[Path] = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to YAML config file"
    ),
):
    """Run deduplication on test data."""
    config = DedupConfig.from_yaml(config_file)
    test_texts = [
        # '    @Test(expected = GadgetException.class)\n    public void malformedGadgetSpecIsCachedAndThrows() throws Exception {\n        HttpRequest request = createCacheableRequest();\n        expect(pipeline.execute(request)).andReturn(new HttpResponse("malformed junk")).once();\n        replay(pipeline);\n        try {\n            specFactory.getGadgetSpec(createContext(SPEC_URL, false));\n            fail("No exception thrown on bad parse");\n        } catch (GadgetException e) {\n        }\n        specFactory.getGadgetSpec(createContext(SPEC_URL, false));\n    }\n',
        # "    public InputStream getInputStream() throws TGBrowserException {\n        try {\n            if (!this.isFolder()) {\n                URL url = new URL(this.url);\n                InputStream stream = url.openStream();\n                return stream;\n            }\n        } catch (Throwable throwable) {\n            throw new TGBrowserException(throwable);\n        }\n        return null;\n    }\n",
        # "    public static void copyFile(File source, File dest) throws IOException {\n        FileChannel in = null, out = null;\n        try {\n            in = new FileInputStream(source).getChannel();\n            out = new FileOutputStream(dest).getChannel();\n            in.transferTo(0, in.size(), out);\n        } catch (FileNotFoundException fnfe) {\n            Log.debug(fnfe);\n        } finally {\n            if (in != null) in.close();\n            if (out != null) out.close();\n        }\n    }\n",
        # '    public static void copyFile(File from, File to) throws IOException {\n        if (from.isDirectory()) {\n            if (!to.exists()) {\n                to.mkdir();\n            }\n            File[] children = from.listFiles();\n            for (int i = 0; i < children.length; i++) {\n                if (children[i].getName().equals(".") || children[i].getName().equals("..")) {\n                    continue;\n                }\n                if (children[i].isDirectory()) {\n                    File f = new File(to, children[i].getName());\n                    copyFile(children[i], f);\n                } else {\n                    copyFile(children[i], to);\n                }\n            }\n        } else if (from.isFile() && (to.isDirectory() || to.isFile())) {\n            if (to.isDirectory()) {\n                to = new File(to, from.getName());\n            }\n            FileInputStream in = new FileInputStream(from);\n            FileOutputStream out = new FileOutputStream(to);\n            byte[] buf = new byte[32678];\n            int read;\n            while ((read = in.read(buf)) > -1) {\n                out.write(buf, 0, read);\n            }\n            closeStream(in);\n            closeStream(out);\n        }\n    }\n',
        "Amrozi accused his brother, whom he called the witness, of deliberately distorting his evidence.	Referring to him as only the witness",
        "Amrozi accused his brother of deliberately distorting his evidence.",
        "Deduplication is so much fun and easy!",
        # "Amrozi accused his brother whom of deliberately distorting witness his evidence."
        # "Hello world...",
        # "Hello",
        # "Hello world?????",
        # "world",
        # "Deduplication is so much fun and easy!",
        # "Python популярный язык для анализа данных",
        # "artificial intelligence will transform healthcare",
        # "ai will transform healthcare and education",
        # "deep blue beat kasparov in 1997",
        # "self driving cars use computer vision",
    ]

    console.print(" Running test deduplication...\n", style="bold")
    strategy = get_strategy(config)

    result = strategy.deduplicate(test_texts)

    _display_results(result, test_texts)


def _display_results(result: dict, texts: list):
    clusters = result["clusters"]
    metadata = result["metadata"]

    table = Table(title=" Deduplication Results")
    table.add_column("Cluster", style="cyan")
    table.add_column("Doc IDs", style="magenta")
    table.add_column("Texts", style="green")

    for cluster_id, doc_ids in sorted(clusters.items()):
        doc_ids_str = ", ".join(map(str, doc_ids))
        texts_str = " | ".join([texts[i] for i in doc_ids])
        table.add_row(str(cluster_id), doc_ids_str, texts_str[:80] + "...")

    console.print(table)
    console.print(" Metadata:")
    console.print(f"   Total texts: {metadata['num_texts']}")
    console.print(f"   Found clusters: {metadata['num_clusters']}")
    console.print(f"   N-gram size: {metadata['ngram_size']}")


@app.command()
def deduplicate_jsonl(
    input_file: Path = typer.Option(None, "--input-file", help="Path to JSONL file"),
    text_field: str = typer.Option(
        "text", "--field", "-f", help="JSON field with text"
    ),
    config_file: Optional[Path] = typer.Option(Path("config.yaml"), "--config", "-c"),
    output_clean: Optional[Path] = typer.Option(
        None, "--clean", help="Path for cleaned JSONL"
    ),
    output_dupes: Optional[Path] = typer.Option(
        None, "--dupes", help="Path for duplicates JSONL"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Directory for output files",
    ),
    display_results: bool = typer.Option(
        False, "--display_results", help="Display clusters in consel"
    ),
    debug: bool = typer.Option(False, "--debug"),
):

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = (
        DedupConfig.from_yaml(config_file) if config_file.exists() else DedupConfig()
    )

    if input_file is None:
        input_file = getattr(config, "input_file", None)
        if isinstance(input_file, str):
            input_file = Path(input_file)

    if not input_file.exists():
        console.print(f"Input file not found: {input_file}", style="red")
        raise typer.Exit(1)
    text_field = getattr(config, "text_field", text_field)

    field_to_read = text_field or getattr(config, "text_field", "text")

    if output_dir is None:
        output_dir = getattr(config, "output_dir", "deduplicate_data")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if output_clean is None:
        output_clean = output_dir / f"clean_{input_file.name}"

    if output_dupes is None:
        output_dupes = output_dir / f"duplicates_{input_file.name}"

    try:
        console.print(f"\nLoading JSONL file: {input_file}")
        console.print(f"Using text field: '{field_to_read}'")
        dataset_objects = []
        texts = []
        with open(input_file, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if text_field in obj:
                        dataset_objects.append(obj)
                        texts.append(str(obj[text_field]))
                except json.JSONDecodeError:
                    continue

        console.print(f"Loaded {len(texts)} documents\n")
        logger.debug(f"Loaded {texts[:5]} ")

        if not texts:
            console.print("No texts extracted from JSONL", style="red")
            raise typer.Exit(1)

        strategy = get_strategy(config)
        result = strategy.deduplicate(texts)
        clusters = result["clusters"]

        console.print(
            "[bold yellow]Separating clean data and duplicates...[/bold yellow]"
        )

        clean_count = 0
        dupes_count = 0

        with open(output_clean, "w") as fc, open(output_dupes, "w") as fd:
            for cluster_id, doc_ids in clusters.items():
                original_idx = doc_ids[0]
                original_obj = dataset_objects[original_idx]

                fc.write(json.dumps(original_obj) + "\n")
                clean_count += 1
                for dup_idx in doc_ids[1:]:
                    dup_obj = dataset_objects[dup_idx]

                    reference_id = original_obj.get("idx", f"line_{original_idx}")
                    dup_obj["__duplicate_of__"] = reference_id
                    dup_obj["__cluster_id__"] = cluster_id

                    fd.write(json.dumps(dup_obj) + "\n")
                    dupes_count += 1
        table = Table(title="Deduplication Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total documents processed", str(len(texts)))
        table.add_row("Unique documents (Clean)", str(clean_count))
        table.add_row("Duplicates removed", str(dupes_count))

        console.print(table)
        console.print(f"\n[green]Clean data saved to:[/green] {output_clean}")
        console.print(f"[green]Duplicates saved to:[/green] {output_dupes}")

        if display_results:
            _display_results(result, texts)

    except Exception as e:
        logger.exception(f"Error: {e}")
        console.print(f"Error: {e}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
