from hsi_plotter.models.style import (
    BUILTIN_PLOT_PRESETS,
    PlotStyle,
    PlotStylePreset,
    plot_style_preset,
)


def test_three_required_plot_presets_are_available() -> None:
    assert set(BUILTIN_PLOT_PRESETS) == {
        PlotStylePreset.PAPER_SINGLE_COLUMN,
        PlotStylePreset.PAPER_DOUBLE_COLUMN,
        PlotStylePreset.PRESENTATION,
    }


def test_plot_presets_return_independent_editable_values() -> None:
    single = plot_style_preset(PlotStylePreset.PAPER_SINGLE_COLUMN)
    double = plot_style_preset(PlotStylePreset.PAPER_DOUBLE_COLUMN)
    presentation = plot_style_preset(PlotStylePreset.PRESENTATION)

    assert isinstance(single, PlotStyle)
    assert single.width_inches < double.width_inches
    assert presentation.font_size > double.font_size
    assert single is not plot_style_preset(PlotStylePreset.PAPER_SINGLE_COLUMN)


def test_plot_style_validates_canvas_and_legend_values() -> None:
    style = PlotStyle(width_inches=7.0, height_inches=3.2, legend_columns=2, dpi=300)

    style.validate()
