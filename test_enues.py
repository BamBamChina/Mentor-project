from dag_modelling.bundles.load_parameters import load_parameters
from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from matplotlib.pyplot import subplots
from numpy import linspace, meshgrid

from dgm_reactor_neutrino import ENuESXsecO1


def test_ENuESXsecO1(debug_graph, test_name: str, output_path: str):
    data = {
        "format": "value",
        "state": "fixed",
        "parameters": {
            "sin_sq_weinberg": 0.235,  # s,   page 165
            "const_fermi": 1.6e-11,  # MeV, page 165
            "ElectronMass": 0.5109989461,  # MeV, page 16
            "PhaseSpaceFactor": 1.71465,
        },
        "labels": {  # TODO:rename labels
            "sin_sq_weinberg": "neutron lifetime, s (PDG2014)",
            "const_fermi": "neutron mass, MeV (PDG2012)",
            "ElectronMass": "electron mass, MeV (PDG2012)",
            "PhaseSpaceFactor": "IBD phase space factor",
        },
    }

    enu1 = linspace(0, 12.0, 11)
    te1 = enu1.copy()
    enu2, te2 = meshgrid(enu1, te1, indexing="ij")

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        storage = load_parameters(data)

        enu = Array("enu", enu2, mode="fill")
        te = Array("te", te2, mode="fill")

        enues_xsec_enu = ENuESXsecO1("ibd_Eν")

        enues_xsec_enu << storage("parameters.constant")

        (enu, te) >> enues_xsec_enu

    csc_enu = enues_xsec_enu.get_data()

    savegraph(graph, f"{output_path}/{test_name}.pdf")

    subplots(1, 1)
    plot_auto(
        csc_enu,
        plotoptions={"method": "pcolormesh"},
        colorbar=True,
        filter_kw={"masked_value": 0},
        show=False,
        close=True,
        save=f"{output_path}/{test_name}_plot.pdf",
    )
