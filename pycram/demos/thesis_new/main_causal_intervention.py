"""
Entry point for controlled causal-intervention experiments.

This keeps the original thesis_new/main.py untouched.

Examples
--------
Create a small PR2/HSRB intervention plan:

    python pycram/demos/thesis_new/main_causal_intervention.py

Run the planned interventions:

    python pycram/demos/thesis_new/main_causal_intervention.py --execute

After the runs, build the paired causal dataset:

    python pycram/demos/thesis_new/main_causal_intervention.py --build-dataset
"""

try:
    from demos.thesis_new.src.causal_intervention_experiment import main
except ImportError:
    try:
        from thesis_new.src.causal_intervention_experiment import main
    except ImportError:
        from src.causal_intervention_experiment import main


if __name__ == "__main__":
    main()
