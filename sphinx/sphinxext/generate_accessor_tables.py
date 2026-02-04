"""
Sphinx extension to automatically generate accessor method tables using Jinja templates.

This extension adds a config value `accessor_summary_classes` which should be a list
of class names to generate accessor summaries for. It automatically generates the
dropdown tables with accessor methods during the build process using a Jinja template.

It also provides a custom ``accessor_dropdown`` directive that renders a 30/70 table
row (class + doc) and a collapsible details/summary block with the same styling as
sphinx-design dropdowns, so the inner content is reliably visible.
"""

import html
import inspect
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from jinja2 import Environment, FileSystemLoader
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom "accessor_dropdown" directive and node (table + details/summary)
# ---------------------------------------------------------------------------


class accessor_dropdown_node(nodes.Element, nodes.General):
    """Node for the accessor dropdown: a table row (class | doc) + details/summary."""

    pass


def visit_accessor_dropdown_html(self, node: accessor_dropdown_node) -> None:
    """Emit <details> with <summary> = one row: [chevron] + table (30/70). Styles in custom.css (sd-acc-*)."""
    name = node.get("name", "")
    doc = (node.get("doc", "") or "").strip().strip('"')
    full_name = node.get("full_name", name)
    uri = node.get("uri", "#")
    anchor = node.get("anchor", full_name)
    opened = node.get("opened", False)
    doc_escaped = html.escape(doc)
    open_attr = ' open="open"' if opened else ""
    class_link = (
        f'<a class="reference internal" href="{html.escape(uri)}#{html.escape(anchor)}" '
        f'title="{html.escape(full_name)}">'
        f'<code class="xref py py-class docutils literal notranslate">'
        f'<span class="pre">{html.escape(name)}</span></code></a>'
    )
    chevron_span = '<span class="sd-acc-chevron" aria-hidden="true"></span>'
    table_html = (
        '<div class="sd-acc-table-wrap">'
        '<table class="table sd-acc-table">'
        "<colgroup><col style=\"width: 30%\"><col style=\"width: 70%\"></colgroup>"
        "<tbody><tr class=\"row-odd\">"
        f"<td>{class_link}</td>"
        f"<td>{doc_escaped}</td>"
        "</tr></tbody></table></div>"
    )
    self.body.append(
        '<div class="sd-sphinx-override sd-dropdown sd-card sd-mb-3 sd-acc-dropdown">'
        f"<details{open_attr}>"
        '<summary class="sd-summary-title sd-card-header sd-acc-summary">'
        f"{chevron_span}{table_html}"
        "</summary>"
        '<div class="sd-summary-content sd-card-body docutils">'
    )


def depart_accessor_dropdown_html(self, node: accessor_dropdown_node) -> None:
    """Close the content div and details."""
    self.body.append("</div></details></div>")


def visit_accessor_dropdown_latex(self, node: accessor_dropdown_node) -> None:
    """LaTeX: emit a paragraph for the title and let children render."""
    name = node.get("name", "")
    doc = node.get("doc", "")
    self.body.append(f"\\paragraph{{{name}}}: {doc}\n\n")


def depart_accessor_dropdown_latex(self, node: accessor_dropdown_node) -> None:
    """LaTeX: nothing to close."""
    pass


class AccessorDropdownDirective(Directive):
    """Directive that renders a table row (class | doc) and a collapsible body with same look as sphinx-design."""

    required_arguments = 1  # class name (short)
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True
    option_spec = {
        "doc": directives.unchanged_required,
        "full_name": directives.unchanged_required,
        "uri": directives.unchanged_required,
        "anchor": directives.unchanged,
        "open": directives.flag,
    }

    def run(self) -> list[nodes.Node]:
        name = self.arguments[0].strip()
        doc = self.options.get("doc", "")
        full_name = self.options.get("full_name", name)
        uri = self.options.get("uri", "#")
        anchor = self.options.get("anchor", full_name)
        opened = "open" in self.options

        node = accessor_dropdown_node(
            "",
            name=name,
            doc=doc,
            full_name=full_name,
            uri=uri,
            anchor=anchor,
            opened=opened,
        )
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def get_doc_first_line(obj):
    doc = getattr(obj, "__doc__", "")
    if doc:
        first_line = doc.strip().split("\n")[0].strip()
        doc = first_line.rstrip(".")
    return doc


def get_accessor_methods(cls: type, accessor_name: str) -> list[tuple[str, str]]:
    """
    Get methods from an accessor by introspecting the accessor class.

    Args:
        cls: The report class (e.g., EstimatorReport)
        accessor_name: The accessor name (e.g., 'metrics')

    Returns:
        List of (method_name, description) tuples
    """
    accessor_cls = getattr(cls, accessor_name)

    if inspect.isclass(accessor_cls):
        # Already a class
        pass
    elif isinstance(accessor_cls, property):
        if accessor_cls.fget is None:
            return []

        sig = inspect.signature(accessor_cls.fget)

        if sig.return_annotation == inspect.Signature.empty:
            logger.debug(f"No return annotation for {cls.__name__}.{accessor_name}")
            return []

        accessor_cls = sig.return_annotation
    else:
        logger.debug(
            f"Unknown accessor type for {cls.__name__}.{accessor_name}: {type(accessor_cls)}"
        )
        return []

    methods = []
    for name in dir(accessor_cls):
        if name.startswith("_"):
            continue
        if name == "help":
            continue  # Skip help method

        attr = getattr(accessor_cls, name)

        if not callable(attr):
            continue

        doc = get_doc_first_line(attr)

        methods.append((name, doc))

    return sorted(methods)


def get_accessor_data(cls: type) -> dict[str, Any]:
    """
    Get accessor data for template rendering.

    Args:
        cls: The class object

    Returns:
        Dictionary with accessor data for template
    """
    if not hasattr(cls, "_ACCESSOR_CONFIG"):
        raise ValueError(f"{cls} has no attribute '_ACCESSOR_CONFIG'.")
    accessor_config: dict = cls._ACCESSOR_CONFIG

    accessors = {}

    for accessor_info in accessor_config.values():
        accessor_name = accessor_info["name"]
        methods = get_accessor_methods(cls, accessor_name) or [
            ("(no public methods)", "")
        ]
        accessors[accessor_name] = {"methods": methods}

    return {
        "name": cls.__name__,
        "doc": get_doc_first_line(cls),
        "accessors": accessors,
    }


def generate_accessor_tables(app: Sphinx, config: Any) -> None:
    """
    Generate accessor table RST files using Jinja template during config-inited event.

    This function reads the `accessor_summary_classes` config value and generates
    RST snippets for each class that can be included in documentation.
    """
    classes_to_process = config.accessor_summary_classes

    if not classes_to_process:
        raise ValueError("accessor_summary_classes not found")

    logger.info("Generating accessor summary tables...")

    classes_data = []

    for class_path in classes_to_process:
        module_name, class_name = class_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)

        class_data = get_accessor_data(cls)
        # Full dotted path for doc links (e.g. skore.EstimatorReport)
        class_data["full_name"] = class_path
        # Relative URI from reference/index to api/ClassName.html
        class_data["uri"] = f"../api/{class_path}.html"
        class_data["anchor"] = class_path
        classes_data.append(class_data)

        logger.info(f"Collected accessor data for {class_name}")

    template_dir = Path(app.confdir) / "_templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("accessor_summary.rst")

    rst_content = template.render(classes=classes_data)

    (Path(app.srcdir) / "reference" / "api").mkdir(exist_ok=True)
    output_path = Path(app.srcdir) / "reference" / "api" / "accessor_tables.rst"
    output_path.write_text(rst_content)
    logger.info(f"Wrote accessor tables to {output_path}")


def setup(app: Sphinx) -> dict[str, Any]:
    """Setup the extension."""
    app.add_config_value("accessor_summary_classes", [], "html")
    app.connect("config-inited", generate_accessor_tables)
    app.add_node(
        accessor_dropdown_node,
        html=(visit_accessor_dropdown_html, depart_accessor_dropdown_html),
        latex=(visit_accessor_dropdown_latex, depart_accessor_dropdown_latex),
    )
    app.add_directive("accessor_dropdown", AccessorDropdownDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
