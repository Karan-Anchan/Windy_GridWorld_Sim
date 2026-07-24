<div align="center">

# WINDY GRIDWORLD // TD CONTROL

**Q-learning, SARSA, and Expected SARSA from scratch on the Sutton & Barto windy gridworld.**

![topic](https://img.shields.io/badge/RL-temporal_difference-7aa2f7?style=flat-square&labelColor=0b0e14)
![python](https://img.shields.io/badge/Python-3.10+-ffb454?style=flat-square&labelColor=0b0e14)
![deps](https://img.shields.io/badge/NumPy_·_Matplotlib-4ec9b0?style=flat-square&labelColor=0b0e14)
![license](https://img.shields.io/badge/license-MIT-8b9080?style=flat-square&labelColor=0b0e14)

</div>

---

This is the **windy gridworld** from Sutton & Barto (Example 6.5), a 10×7 grid with a column-dependent wind that pushes the agent upward. The agent starts at `(0, 3)`, receives **−1 per step**, and stops when it reaches `(7, 3)`. The repository compares three tabular temporal-difference control methods:

| Method | Backup target | On/off-policy |
| :--- | :--- | :---: |
| Q-learning | `max_a' Q(s', a')` | off-policy |
| SARSA | `Q(s', a')` for the actually-taken `a'` | on-policy |
| Expected SARSA | `E_π[Q(s', a')]` under the ε-greedy policy | on-policy |

## Results

<p align="center">
<img src="images/learning_curves.png" width="52%" alt="Learning curves for the three methods"/>
&nbsp;
<img src="images/agent_path.gif" width="44%" alt="Trained agent walking the windy grid to the goal"/>
</p>

- All three methods learn a near-optimal policy. The greedy path reaches the goal in **17 steps** by moving above the goal row and accounting for the wind.
- Q-learning has the highest cumulative reward. Expected SARSA produces the smoothest curve because its backup has lower variance. The curves are averaged over 300 independent runs.

## Reproduce

```bash
pip install numpy matplotlib
python main.py    # trains all three methods -> images/learning_curves.png + agent_path.gif
```

Grid size, wind profile, start/goal, and `episodes`/`trials` are set at the top of `main.py`.

## Repository layout

```
windy_gridworld.py  environment: wind dynamics, step, reset
algorithms.py       q_learning / sarsa / expected_sarsa + train_q_table
visualizations.py   learning-curve plot + greedy-path animation
main.py             train the three methods and render the figures
```

## Limitations

The environment uses the standard four actions. It does not include diagonal moves or stochastic wind. Both ε and α remain fixed, so the averaged curves settle slightly below the deterministic optimum.

## References

Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Example 6.5 · [book](http://incompleteideas.net/book/the-book.html)

MIT. See [`LICENSE`](LICENSE).
