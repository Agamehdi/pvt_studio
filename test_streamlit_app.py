from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).with_name("app.py")


def test_input_unit_change_converts_value_and_waits_for_run():
    app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
    unit = next(widget for widget in app.selectbox if widget.label == "Minimum pressure unit")
    unit.select("bar(a)")
    app.run()
    converted = next(widget for widget in app.number_input if widget.label == "Minimum pressure")
    assert np.isclose(converted.value, 6.894757293178307)
    assert any("click **Run PVT**" in message.value for message in app.info)


@pytest.mark.parametrize("fluid", ["Black Oil", "Dead Oil", "Dry Gas", "Wet Gas"])
def test_streamlit_app_runs_each_fluid_mode(fluid):
    app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
    assert not app.exception

    fluid_selector = next(widget for widget in app.selectbox if widget.label == "Fluid system")
    if fluid != "Black Oil":
        fluid_selector.select(fluid)
        app.run()
        assert not app.exception

    run_button = next(widget for widget in app.button if widget.label == "▶ Run PVT")
    run_button.click()
    app.run(timeout=45)
    assert not app.exception
    assert any(item.value == "Key results" for item in app.subheader)
    assert [item.label for item in app.get("download_button")] == [
        "Download CSV",
        "Download ASCII",
        "Download Eclipse (FIELD)",
    ]
