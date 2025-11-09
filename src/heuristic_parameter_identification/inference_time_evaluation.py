from time import perf_counter_ns
import numpy as np
import pysmile_license # pysmile license required in the same directory of this python script
import pysmile
from pysmile import Network
from scipy.stats import bootstrap

def computePerformanceAnalysis():
    ######################################### CONFIGURATION ############################################################
    xdsl_path = "../../bayesian_networks/bayesian_network_based_on_fmeca_data.xdsl" # path to XDSL file of the BN
    n_repeats = 100  # Number of repetitions for statistical analysis

    # === Import the BN
    net = Network()
    net.read_file(xdsl_path)

    ################################# Define the 20 evidence sets of the case-based test ###############################
    evidence_sets = [
        {"F01_Greasing": "FM01_Grease_incorrectly_dispensed"}, # 1
        {"F02_Feeding_cartridges": "FM02_Cartridge_is_damaged"}, # 2
        {"F03_Positioning_cartridges": "FM03_Not_pressed_in_position", "FC04_Insufficient_accuracy_of_the_axes": "False"}, # 3
        {"F03_Positioning_cartridges": "FM04_Surface_of_cartridge_bore_damaged"}, # 4
        {"FC01_Air_inclusions_in_the_grease": "True"}, #5
        {"FC04_Insufficient_accuracy_of_the_axes": "True"}, #6
        {"FC04_Insufficient_accuracy_of_the_axes": "True"}, #7
        {"FC05_Press_in_force_too_low": "True"}, #8
        {"FC08_Incorrect_press_settings": "True"}, #9
        {"FC03_Position_of_the_cartridge_on_the_pressing_in_plug_imprecise": "True", "FE05_Control_head_is_pressurised": "True"}, #10
        {"FC05_Press_in_force_too_low": "True", "FE07_Failure_valve_no_switching_of_air": "True"}, #11
        {"FC06_Incorrect_orientation_of_the_cartridge": "True"}, #12
        {"FE01_Valve_switching_times_not_reached": "True"}, #13
        {"FE05_Control_head_is_pressurised": "True", "F02_Feeding_cartridges": "UNK"}, #14
        {"FE04_Internal_leakage": "True", "FC04_Insufficient_accuracy_of_the_axes": "False"}, #15
        {"FE06_Control_head_not_controlled": "True", "FC04_Insufficient_accuracy_of_the_axes": "False", "FE03_FC07_Cartridge_bore_not_completely_greased": "False", "FC08_Incorrect_press_settings": "False"}, #16
        {"F01_Greasing": "FM01_Grease_incorrectly_dispensed"}, #17
        {"F02_Feeding_cartridges": "FM02_Cartridge_is_damaged"}, #18
        {"F01_Greasing": "FM01_Grease_incorrectly_dispensed", "F03_Positioning_cartridges": "OK"}, #19
        {"FC03_Position_of_the_cartridge_on_the_pressing_in_plug_imprecise": "True"}, #20
    ]

    ################################# Computational performance analysis ###############################################
    all_timings = {}

    # loop to perform the inference
    for repeat in range(n_repeats):
        run_timings = []
        for evidence in evidence_sets:
            # clear all evidence
            net.clear_all_evidence()

            # set evidence
            for node, value in evidence.items():
                net.set_evidence(node, value)

            start_time = perf_counter_ns() * 10**-6
            net.update_beliefs()
            end_time = perf_counter_ns() * 10**-6

            delta_time = end_time - start_time
            run_timings.append(delta_time)

        all_timings[f"iteration_{repeat}"] = run_timings

    # Statistical summary dictionary
    statistics_summary = {}

    for i, evidence in enumerate(evidence_sets):
        timings = [run[i] for run in all_timings.values()]

        # Statistical metrics
        mean_time = np.mean(timings)
        std_dev = np.std(timings)
        lower_percentile = np.percentile(timings, 2.5)
        upper_percentile = np.percentile(timings, 97.5)

        # Bootstrap CI
        data = (np.array(timings),)
        ci_result = bootstrap(
            data,
            np.mean,
            confidence_level=0.95,
            n_resamples=1000,
            method='percentile',
            random_state=42
        )
        ci_low, ci_high = ci_result.confidence_interval

        statistics_summary[f"Evidence_Set_{i+1}"] = {
            "Evidence": evidence,
            "Mean": mean_time,
            "StdDev": std_dev,
            "95%_Percentile_Range": [lower_percentile, upper_percentile],
            "95%_Bootstrap_CI": [ci_low, ci_high],
            "All_Timings": timings
        }

    # total inference time across all 20 sample fault cases
    statistics_summary["dict_total_inference_time"] = {}
    for i, (_, timing) in enumerate(all_timings.items()):
        statistics_summary["dict_total_inference_time"][f"iteration_{i}"] = {
            "Mean": np.mean(timing),
            "StdDev": np.std(timing),
            "95%_Percentile_Range": [np.percentile(timing, 2.5), np.percentile(timing, 97.5)],
            "All_Timings": timing
        }

    # Bootstrap CI
    means = ([v["Mean"] for _, v in statistics_summary["dict_total_inference_time"].items()],)
    ci_result = bootstrap(
        means,
        np.mean,
        confidence_level=0.95,
        n_resamples=1000,
        method='percentile',
        random_state=42
    )
    ci_low, ci_high = ci_result.confidence_interval

    statistics_summary["dict_total_inference_time"]["total_inference_time_statistics"] = {
            "Mean": np.mean(means),
            "StdDev": np.std(means),
            "95%_Bootstrap_CI": [ci_low, ci_high],
            "all_means": means
    }

    print(statistics_summary)

    ##################################### Print summary for verification ###############################################
    for key, stats in statistics_summary.items():
        if key != "dict_total_inference_time":
            print(f"\n{key}:")
            print(f"  Mean: {stats['Mean']:.6f}s")
            print(f"  Std Dev: {stats['StdDev']:.6f}s")
            print(f"  95% Percentile Range: [{stats['95%_Percentile_Range'][0]:.6f}s, {stats['95%_Percentile_Range'][1]:.6f}s]")
            print(f"  95% Bootstrap CI: [{stats['95%_Bootstrap_CI'][0]:.6f}s, {stats['95%_Bootstrap_CI'][1]:.6f}s]")
        else:
            print(f"\n{key.upper()}:")
            print(f"  Mean: {stats['total_inference_time_statistics']['Mean']:.6f}s")
            print(f"  Std Dev: {stats['total_inference_time_statistics']['StdDev']:.6f}s")
            print(f"  95% Bootstrap CI: [{stats['total_inference_time_statistics']['95%_Bootstrap_CI'][0]:.6f}s, {stats['total_inference_time_statistics']['95%_Bootstrap_CI'][1]:.6f}s]")
            stats['total_inference_time_statistics']



if __name__ == "__main__":
    computePerformanceAnalysis()