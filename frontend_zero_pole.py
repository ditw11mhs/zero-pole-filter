import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame as df
from backend_zero_pole import ZeroPoleFilter


class Main(ZeroPoleFilter):
    def __init__(self):
        pass

    def main(self):

        st.set_page_config(
            page_title="Zero Pole Filter",
            page_icon="⭕",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.title("LPF Zero Pole Filter")
        c1, c2 = st.columns(2)

        with c1.expander("LPF H(z)"):
            st.subheader("LPF H(z)")
            st.latex(
                r"H(z) = \frac{(z+1)(z+1)}{(z-re^{\mathit{j\theta}})(z-re^{\mathit{-j\theta}})}"
            )
            st.latex(
                r"= \frac{(z+1)(z+1)}{z^{2}-zr(e^{\mathit{j\theta}}+e^\mathit{-j\theta})+r^2}"
            )
            st.latex(
                r"= \frac{z^2+2z+1}{z^{2}-zr(cos\theta+jsin\theta+cos\theta-jsin\theta)+r^2}"
            )
            st.latex(r"= \frac{z^2+2z+1}{z^{2}-2zrcos\theta+r^2}")
            st.latex(r"= \frac{1+2z^{-1}+z^{-2}}{1-2rcos\theta z^{-1}+r^{2}z^{-2}}")

        with c1.expander("Filter Equation"):
            st.subheader("Filter Equation")
            st.latex(r"y[n]=x[n]+2x[n-1]+x[n-2]+2rcos{\theta}y[n-1]-r^{2}y[n-2]")

        with c2.expander("LPF Frequency Response"):
            st.subheader("LPF Frequency Response")
            st.latex(r"H(\Omega )=H(z)|z=e^{j\Omega}")
            st.latex(
                r"=\frac{1+2z^{-1}+z^{-2}}{1-2rcos\theta z^{-1}+r^{2}z^{-2}}|z=e^{j\Omega}"
            )
            st.latex(
                r"=\frac{1+2(cos\Omega-jsin\Omega)+cos2\Omega-jsin2\Omega}{1-2rcos\theta (cos\Omega-jsin\Omega)+r^{2}(cos2\Omega-jsin2\Omega)}"
            )
            st.latex(
                r"=\frac{1+2cos\Omega+cos2\Omega-j(2sin\Omega+sin2\Omega)}{1-2rcos\theta cos\Omega+ r^{2}cos{2\Omega}+j(2rcos{\theta}sin{\Omega}-r^{2}sin{2\Omega})}"
            )
            st.latex(
                r"H|(\Omega)|=\frac{\sqrt{(1+2cos\Omega+cos2\Omega)^{2}+(-2sin\Omega-sin2\Omega)^{2}}}{\sqrt{(1-2rcos\theta cos\Omega+ r^{2}cos{2\Omega})^{2}+(2rcos{\theta}sin{\Omega}-r^{2}sin{2\Omega})^{2}}}"
            )

        with c2.expander("LPF Phase Response"):
            st.subheader("LPF Phase Response")
            st.latex(
                r"\phi=arctan\frac{-2sin\Omega-sin2\Omega}{1+2cos\Omega+cos2\Omega}-arctan\frac{2rcos{\theta}sin{\Omega}-r^{2}sin{2\Omega}}{1-2rcos\theta cos\Omega+ r^{2}cos{2\Omega}}"
            )

        form_container = st.sidebar.container()
        wave_state, filter_state = self._create_form(form_container)
        self.plot_color = "cyan"
        if wave_state["Submitted"]:
            raw_waves = self.wave_gen(wave_state)
            self.plot_waves(raw_waves, c1)

            raw_waves_dft = self.dft(
                raw_waves["Waves"], wave_state["Sampling Frequency"]
            )
            self.plot_raw_waves_dft(raw_waves_dft, c2)
            (
                filtered_waves,
                filtered_waves_dft,
                pole_zero,
                filter_omega,
                filter_phi
            ) = self.filter_waves(
                raw_waves["Waves"], filter_state, wave_state["Sampling Frequency"]
            )
            # st.write(filtered_waves)
            self.plot_filtered_waves(filtered_waves, c1)
            self.plot_filtered_waves_dft(filtered_waves_dft, c2)
            self.plot_filter_omega(filter_omega, c2)
            self.plot_pole_zero(pole_zero, c1)
            self.plot_filter_phi(filter_phi, c1)
    
    def plot_filter_phi(self, filter_phi, container):
        container.subheader("Phase Response")
        filter_phi_df = df(
            {
                "Phase": filter_phi["Phase"],
                "Frequency": filter_phi["Frequency"],
            }
        )
        filter_phi_fig = px.line(
            filter_phi_df,
            x="Frequency",
            y="Phase",
            color_discrete_sequence=[self.plot_color],
        )
        filter_phi_fig.update_layout(yaxis_title="Phase")
        container.plotly_chart(filter_phi_fig, use_container_width=True)
        
    def plot_pole_zero(self, pole_zero, container):
        container.subheader("Filter Pole Zero")
        pole_zero_df = df(
            {
                "X Axis": pole_zero["X Axis"],
                "Y Axis": pole_zero["Y Axis"],
                "Type": pole_zero["Type"],
            }
        )
        pole_zero_fig = go.Figure()
        pole_zero_fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=-1,
            y0=-1,
            x1=1,
            y1=1,
            line={"width": 3, "color": self.plot_color},
        )
        pole_zero_fig.add_trace(
            go.Scatter(
                mode="markers",
                x=pole_zero_df["X Axis"],
                y=pole_zero_df["Y Axis"],
                marker_symbol=["circle-open", "circle-open", "x", "x"],
                marker_size=15,
                marker_color=pole_zero_df["Type"],
            )
        )
        pole_zero_fig.update_layout(yaxis_title="Gain")
        container.plotly_chart(pole_zero_fig, use_container_width=True)

    def plot_filter_omega(self, filter_omega, container):
        container.subheader("Filter Gain")
        filter_omega_df = df(
            {
                "Gain": filter_omega["Gain"],
                "Frequency": filter_omega["Frequency"],
            }
        )
        filter_omega_fig = px.line(
            filter_omega_df,
            x="Frequency",
            y="Gain",
            color_discrete_sequence=[self.plot_color],
        )
        filter_omega_fig.update_layout(yaxis_title="Gain")
        container.plotly_chart(filter_omega_fig, use_container_width=True)

    def plot_filtered_waves_dft(self, filtered_waves_dft, container):
        container.subheader("Filtered Waves DFT")
        filtered_waves_dft_df = df(
            {
                "Amplitude": filtered_waves_dft["Amplitude"],
                "Frequency": filtered_waves_dft["Frequency"],
            }
        )
        filtered_waves_dft_fig = px.bar(
            filtered_waves_dft_df,
            x="Frequency",
            y="Amplitude",
            color_discrete_sequence=[self.plot_color],
        )
        filtered_waves_dft_fig.update_layout(yaxis_title="Amplitude")
        container.plotly_chart(filtered_waves_dft_fig, use_container_width=True)

    def plot_filtered_waves(self, filtered_waves, container):
        container.subheader("Filtered Waves")
        filtered_waves_df = df(
            {
                "Filtered Waves": filtered_waves["Filtered Waves"],
                "Time": filtered_waves["Time"],
            }
        )
        filtered_waves_fig = px.line(
            filtered_waves_df,
            x="Time",
            y="Filtered Waves",
            color_discrete_sequence=[self.plot_color],
        )
        filtered_waves_fig.update_layout(yaxis_title="Amplitude")
        container.plotly_chart(filtered_waves_fig, use_container_width=True)

    def plot_waves(self, raw_waves, container):
        container.subheader("Raw Waves")
        raw_waves_df = df(
            {
                "Waves": raw_waves["Waves"],
                "Time": raw_waves["t"],
            },
        )
        raw_waves_fig = px.line(
            raw_waves_df, x="Time", y="Waves", color_discrete_sequence=[self.plot_color]
        )
        raw_waves_fig.update_layout(yaxis_title="Amplitude")
        container.plotly_chart(raw_waves_fig, use_container_width=True)

    def plot_raw_waves_dft(self, raw_waves_dft, container):
        container.subheader("Raw Waves DFT")
        raw_waves_dft_df = df(
            {
                "Amplitude": raw_waves_dft["Amplitude"],
                "Frequency": raw_waves_dft["Frequency"],
            },
        )
        raw_waves_dft_fig = px.bar(
            raw_waves_dft_df,
            x="Frequency",
            y="Amplitude",
            color_discrete_sequence=[self.plot_color],
        )
        raw_waves_dft_fig.update_layout(
            xaxis_title="Frequency (hz)", yaxis_title="Magnitude"
        )
        container.plotly_chart(raw_waves_dft_fig, use_container_width=True)

    def _create_form(self, container):
        wave_form = container.form(key="Wave Form")
        wave_form.header("Parameter")
        wave_form.subheader("Wave Parameters")
        wave_state = {}
        for num in np.arange(1, 4):
            wave_state["Wave " + str(num)] = {}
            wave_state["Wave " + str(num)]["Frequency"] = wave_form.number_input(
                "Frequency Wave " + str(num),
                min_value=0,
                max_value=1000,
                value=num * 10,
            )
            wave_state["Wave " + str(num)]["Amplitude"] = wave_form.number_input(
                "Amplitude Wave " + str(num),
                min_value=0,
                max_value=1000,
                value=num * 3,
            )
        wave_state["Sampling Frequency"] = wave_form.number_input(
            "Sampling Frequency",
            min_value=0,
            max_value=50000,
            value=1000,
        )
        wave_state["Duration"] = wave_form.number_input(
            "Duration", min_value=0, max_value=5, value=1
        )
        wave_form.subheader("Noise Parameters")
        wave_state["Standard Deviation"] = wave_form.number_input(
            "Standard Deviation", min_value=0.0, max_value=10.0, value=0.0
        )
        wave_state["Mean"] = wave_form.number_input(
            "Mean", min_value=0.0, max_value=10.0, value=0.5
        )

        wave_form.header("Filter Parameter")
        filter_state = {}
        filter_state["Pole Radius"] = wave_form.number_input(
            "Pole Radius", min_value=0.0, max_value=1.0, value=0.5
        )
        filter_state["Zero Radius"] = 1
        filter_state["Cutoff Frequency"] = wave_form.number_input(
            "Cutoff Frequency", min_value=0, max_value=1000, value=10
        )
        wave_state["Submitted"] = wave_form.form_submit_button("Get Waves")
        return wave_state, filter_state


if __name__ == "__main__":
    main = Main()
    main.main()
