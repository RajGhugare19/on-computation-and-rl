# pointmaze-medium-navigate-v0 (GCBC)
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# pointmaze-medium-navigate-v0 (GCIVL)
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# pointmaze-medium-navigate-v0 (GCIQL)
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.003
# pointmaze-medium-navigate-v0 (QRL)
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.0003
# pointmaze-medium-navigate-v0 (CRL)
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.03
# pointmaze-medium-navigate-v0 (HIQL)
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0

# pointmaze-large-navigate-v0 (GCBC)
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# pointmaze-large-navigate-v0 (GCIVL)
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# pointmaze-large-navigate-v0 (GCIQL)
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.003
# pointmaze-large-navigate-v0 (QRL)
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.0003
# pointmaze-large-navigate-v0 (CRL)
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.03
# pointmaze-large-navigate-v0 (HIQL)
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0

# pointmaze-giant-navigate-v0 (GCBC)
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# pointmaze-giant-navigate-v0 (GCIVL)
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# pointmaze-giant-navigate-v0 (GCIQL)
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.003 --agent.discount=0.995
# pointmaze-giant-navigate-v0 (QRL)
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.0003 --agent.discount=0.995
# pointmaze-giant-navigate-v0 (CRL)
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.03 --agent.discount=0.995
# pointmaze-giant-navigate-v0 (HIQL)
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# pointmaze-teleport-navigate-v0 (GCBC)
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# pointmaze-teleport-navigate-v0 (GCIVL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# pointmaze-teleport-navigate-v0 (GCIQL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.003
# pointmaze-teleport-navigate-v0 (QRL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.0003
# pointmaze-teleport-navigate-v0 (CRL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.03
# pointmaze-teleport-navigate-v0 (HIQL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0

# pointmaze-medium-stitch-v0 (GCBC)
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# pointmaze-medium-stitch-v0 (GCIVL)
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# pointmaze-medium-stitch-v0 (GCIQL)
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# pointmaze-medium-stitch-v0 (QRL)
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.0003
# pointmaze-medium-stitch-v0 (CRL)
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.03
# pointmaze-medium-stitch-v0 (HIQL)
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# pointmaze-large-stitch-v0 (GCBC)
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# pointmaze-large-stitch-v0 (GCIVL)
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# pointmaze-large-stitch-v0 (GCIQL)
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# pointmaze-large-stitch-v0 (QRL)
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.0003
# pointmaze-large-stitch-v0 (CRL)
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.03
# pointmaze-large-stitch-v0 (HIQL)
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# pointmaze-giant-stitch-v0 (GCBC)
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# pointmaze-giant-stitch-v0 (GCIVL)
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.discount=0.995
# pointmaze-giant-stitch-v0 (GCIQL)
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.discount=0.995
# pointmaze-giant-stitch-v0 (QRL)
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.0003 --agent.discount=0.995
# pointmaze-giant-stitch-v0 (CRL)
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.03 --agent.discount=0.995
# pointmaze-giant-stitch-v0 (HIQL)
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# pointmaze-teleport-stitch-v0 (GCBC)
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# pointmaze-teleport-stitch-v0 (GCIVL)
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# pointmaze-teleport-stitch-v0 (GCIQL)
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# pointmaze-teleport-stitch-v0 (QRL)
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.0003
# pointmaze-teleport-stitch-v0 (CRL)
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.03
# pointmaze-teleport-stitch-v0 (HIQL)
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-medium-navigate-v0 (GCBC)
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-medium-navigate-v0 (GCIVL)
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# antmaze-medium-navigate-v0 (GCIQL)
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3
# antmaze-medium-navigate-v0 (QRL)
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.003
# antmaze-medium-navigate-v0 (CRL)
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# antmaze-medium-navigate-v0 (HIQL)
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-large-navigate-v0 (GCBC)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-large-navigate-v0 (GCIVL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# antmaze-large-navigate-v0 (GCIQL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3
# antmaze-large-navigate-v0 (QRL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.003
# antmaze-large-navigate-v0 (CRL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# antmaze-large-navigate-v0 (HIQL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-giant-navigate-v0 (GCBC)
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-giant-navigate-v0 (GCIVL)
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# antmaze-giant-navigate-v0 (GCIQL)
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3 --agent.discount=0.995
# antmaze-giant-navigate-v0 (QRL)
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.003 --agent.discount=0.995
# antmaze-giant-navigate-v0 (CRL)
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# antmaze-giant-navigate-v0 (HIQL)
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-teleport-navigate-v0 (GCBC)
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-teleport-navigate-v0 (GCIVL)
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# antmaze-teleport-navigate-v0 (GCIQL)
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3
# antmaze-teleport-navigate-v0 (QRL)
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.003
# antmaze-teleport-navigate-v0 (CRL)
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# antmaze-teleport-navigate-v0 (HIQL)
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-medium-stitch-v0 (GCBC)
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-medium-stitch-v0 (GCIVL)
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# antmaze-medium-stitch-v0 (GCIQL)
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
# antmaze-medium-stitch-v0 (QRL)
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# antmaze-medium-stitch-v0 (CRL)
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
# antmaze-medium-stitch-v0 (HIQL)
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-large-stitch-v0 (GCBC)
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-large-stitch-v0 (GCIVL)
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# antmaze-large-stitch-v0 (GCIQL)
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
# antmaze-large-stitch-v0 (QRL)
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# antmaze-large-stitch-v0 (CRL)
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
# antmaze-large-stitch-v0 (HIQL)
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-giant-stitch-v0 (GCBC)
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-giant-stitch-v0 (GCIVL)
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.discount=0.995
# antmaze-giant-stitch-v0 (GCIQL)
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.discount=0.995
# antmaze-giant-stitch-v0 (QRL)
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.discount=0.995
# antmaze-giant-stitch-v0 (CRL)
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
# antmaze-giant-stitch-v0 (HIQL)
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-teleport-stitch-v0 (GCBC)
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-teleport-stitch-v0 (GCIVL)
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# antmaze-teleport-stitch-v0 (GCIQL)
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
# antmaze-teleport-stitch-v0 (QRL)
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# antmaze-teleport-stitch-v0 (CRL)
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
# antmaze-teleport-stitch-v0 (HIQL)
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-medium-explore-v0 (GCBC)
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-medium-explore-v0 (GCIVL)
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0
# antmaze-medium-explore-v0 (GCIQL)
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01
# antmaze-medium-explore-v0 (QRL)
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001
# antmaze-medium-explore-v0 (CRL)
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.003
# antmaze-medium-explore-v0 (HIQL)
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# antmaze-large-explore-v0 (GCBC)
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-large-explore-v0 (GCIVL)
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0
# antmaze-large-explore-v0 (GCIQL)
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01
# antmaze-large-explore-v0 (QRL)
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001
# antmaze-large-explore-v0 (CRL)
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.003
# antmaze-large-explore-v0 (HIQL)
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# antmaze-teleport-explore-v0 (GCBC)
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antmaze-teleport-explore-v0 (GCIVL)
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0
# antmaze-teleport-explore-v0 (GCIQL)
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01
# antmaze-teleport-explore-v0 (QRL)
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001
# antmaze-teleport-explore-v0 (CRL)
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.003
# antmaze-teleport-explore-v0 (HIQL)
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# humanoidmaze-medium-navigate-v0 (GCBC)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# humanoidmaze-medium-navigate-v0 (GCIVL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-medium-navigate-v0 (GCIQL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-medium-navigate-v0 (QRL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-medium-navigate-v0 (CRL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-medium-navigate-v0 (HIQL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100

# humanoidmaze-large-navigate-v0 (GCBC)
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# humanoidmaze-large-navigate-v0 (GCIVL)
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-large-navigate-v0 (GCIQL)
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-large-navigate-v0 (QRL)
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-large-navigate-v0 (CRL)
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-large-navigate-v0 (HIQL)
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100

# humanoidmaze-giant-navigate-v0 (GCBC)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# humanoidmaze-giant-navigate-v0 (GCIVL)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-giant-navigate-v0 (GCIQL)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-giant-navigate-v0 (QRL)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-giant-navigate-v0 (CRL)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-giant-navigate-v0 (HIQL)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100

# humanoidmaze-medium-stitch-v0 (GCBC)
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# humanoidmaze-medium-stitch-v0 (GCIVL)
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-medium-stitch-v0 (GCIQL)
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-medium-stitch-v0 (QRL)
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-medium-stitch-v0 (CRL)
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-medium-stitch-v0 (HIQL)
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100

# humanoidmaze-large-stitch-v0 (GCBC)
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# humanoidmaze-large-stitch-v0 (GCIVL)
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-large-stitch-v0 (GCIQL)
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-large-stitch-v0 (QRL)
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-large-stitch-v0 (CRL)
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-large-stitch-v0 (HIQL)
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100

# humanoidmaze-giant-stitch-v0 (GCBC)
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# humanoidmaze-giant-stitch-v0 (GCIVL)
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-giant-stitch-v0 (GCIQL)
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-giant-stitch-v0 (QRL)
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-giant-stitch-v0 (CRL)
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-giant-stitch-v0 (HIQL)
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100

# antsoccer-arena-navigate-v0 (GCBC)
python main.py --env_name=antsoccer-arena-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antsoccer-arena-navigate-v0 (GCIVL)
python main.py --env_name=antsoccer-arena-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# antsoccer-arena-navigate-v0 (GCIQL)
python main.py --env_name=antsoccer-arena-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1
# antsoccer-arena-navigate-v0 (QRL)
python main.py --env_name=antsoccer-arena-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.003
# antsoccer-arena-navigate-v0 (CRL)
python main.py --env_name=antsoccer-arena-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.3
# antsoccer-arena-navigate-v0 (HIQL)
python main.py --env_name=antsoccer-arena-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antsoccer-medium-navigate-v0 (GCBC)
python main.py --env_name=antsoccer-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antsoccer-medium-navigate-v0 (GCIVL)
python main.py --env_name=antsoccer-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# antsoccer-medium-navigate-v0 (GCIQL)
python main.py --env_name=antsoccer-medium-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1
# antsoccer-medium-navigate-v0 (QRL)
python main.py --env_name=antsoccer-medium-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.003
# antsoccer-medium-navigate-v0 (CRL)
python main.py --env_name=antsoccer-medium-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.3
# antsoccer-medium-navigate-v0 (HIQL)
python main.py --env_name=antsoccer-medium-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antsoccer-arena-stitch-v0 (GCBC)
python main.py --env_name=antsoccer-arena-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antsoccer-arena-stitch-v0 (GCIVL)
python main.py --env_name=antsoccer-arena-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# antsoccer-arena-stitch-v0 (GCIQL)
python main.py --env_name=antsoccer-arena-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
# antsoccer-arena-stitch-v0 (QRL)
python main.py --env_name=antsoccer-arena-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# antsoccer-arena-stitch-v0 (CRL)
python main.py --env_name=antsoccer-arena-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
# antsoccer-arena-stitch-v0 (HIQL)
python main.py --env_name=antsoccer-arena-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antsoccer-medium-stitch-v0 (GCBC)
python main.py --env_name=antsoccer-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
# antsoccer-medium-stitch-v0 (GCIVL)
python main.py --env_name=antsoccer-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# antsoccer-medium-stitch-v0 (GCIQL)
python main.py --env_name=antsoccer-medium-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
# antsoccer-medium-stitch-v0 (QRL)
python main.py --env_name=antsoccer-medium-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# antsoccer-medium-stitch-v0 (CRL)
python main.py --env_name=antsoccer-medium-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
# antsoccer-medium-stitch-v0 (HIQL)
python main.py --env_name=antsoccer-medium-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# visual-antmaze-medium-navigate-v0 (GCBC)
python main.py --env_name=visual-antmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-navigate-v0 (GCIVL)
python main.py --env_name=visual-antmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-navigate-v0 (GCIQL)
python main.py --env_name=visual-antmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-navigate-v0 (QRL)
python main.py --env_name=visual-antmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-navigate-v0 (CRL)
python main.py --env_name=visual-antmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-navigate-v0 (HIQL)
python main.py --env_name=visual-antmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-large-navigate-v0 (GCBC)
python main.py --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-navigate-v0 (GCIVL)
python main.py --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-navigate-v0 (GCIQL)
python main.py --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-navigate-v0 (QRL)
python main.py --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-navigate-v0 (CRL)
python main.py --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-navigate-v0 (HIQL)
python main.py --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-giant-navigate-v0 (GCBC)
python main.py --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-giant-navigate-v0 (GCIVL)
python main.py --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-navigate-v0 (GCIQL)
python main.py --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.3 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-navigate-v0 (QRL)
python main.py --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.003 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-navigate-v0 (CRL)
python main.py --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-navigate-v0 (HIQL)
python main.py --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-teleport-navigate-v0 (GCBC)
python main.py --env_name=visual-antmaze-teleport-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-navigate-v0 (GCIVL)
python main.py --env_name=visual-antmaze-teleport-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-navigate-v0 (GCIQL)
python main.py --env_name=visual-antmaze-teleport-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-navigate-v0 (QRL)
python main.py --env_name=visual-antmaze-teleport-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-navigate-v0 (CRL)
python main.py --env_name=visual-antmaze-teleport-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-navigate-v0 (HIQL)
python main.py --env_name=visual-antmaze-teleport-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-medium-stitch-v0 (GCBC)
python main.py --env_name=visual-antmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-stitch-v0 (GCIVL)
python main.py --env_name=visual-antmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-stitch-v0 (GCIQL)
python main.py --env_name=visual-antmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-stitch-v0 (QRL)
python main.py --env_name=visual-antmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-stitch-v0 (CRL)
python main.py --env_name=visual-antmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-stitch-v0 (HIQL)
python main.py --env_name=visual-antmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-large-stitch-v0 (GCBC)
python main.py --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-stitch-v0 (GCIVL)
python main.py --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-stitch-v0 (GCIQL)
python main.py --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-stitch-v0 (QRL)
python main.py --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-stitch-v0 (CRL)
python main.py --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-stitch-v0 (HIQL)
python main.py --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-giant-stitch-v0 (GCBC)
python main.py --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-giant-stitch-v0 (GCIVL)
python main.py --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-stitch-v0 (GCIQL)
python main.py --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-stitch-v0 (QRL)
python main.py --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-stitch-v0 (CRL)
python main.py --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-stitch-v0 (HIQL)
python main.py --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-teleport-stitch-v0 (GCBC)
python main.py --env_name=visual-antmaze-teleport-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-stitch-v0 (GCIVL)
python main.py --env_name=visual-antmaze-teleport-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-stitch-v0 (GCIQL)
python main.py --env_name=visual-antmaze-teleport-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-stitch-v0 (QRL)
python main.py --env_name=visual-antmaze-teleport-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-stitch-v0 (CRL)
python main.py --env_name=visual-antmaze-teleport-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-stitch-v0 (HIQL)
python main.py --env_name=visual-antmaze-teleport-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-medium-explore-v0 (GCBC)
python main.py --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-explore-v0 (GCIVL)
python main.py --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-explore-v0 (GCIQL)
python main.py --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-explore-v0 (QRL)
python main.py --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-explore-v0 (CRL)
python main.py --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-explore-v0 (HIQL)
python main.py --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=10.0 --agent.low_actor_rep_grad=True --agent.low_alpha=10.0

# visual-antmaze-large-explore-v0 (GCBC)
python main.py --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-explore-v0 (GCIVL)
python main.py --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-explore-v0 (GCIQL)
python main.py --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-explore-v0 (QRL)
python main.py --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-explore-v0 (CRL)
python main.py --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-explore-v0 (HIQL)
python main.py --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=10.0 --agent.low_actor_rep_grad=True --agent.low_alpha=10.0

# visual-antmaze-teleport-explore-v0 (GCBC)
python main.py --env_name=visual-antmaze-teleport-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-explore-v0 (GCIVL)
python main.py --env_name=visual-antmaze-teleport-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-explore-v0 (GCIQL)
python main.py --env_name=visual-antmaze-teleport-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-explore-v0 (QRL)
python main.py --env_name=visual-antmaze-teleport-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-explore-v0 (CRL)
python main.py --env_name=visual-antmaze-teleport-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-teleport-explore-v0 (HIQL)
python main.py --env_name=visual-antmaze-teleport-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=10.0 --agent.low_actor_rep_grad=True --agent.low_alpha=10.0

# visual-humanoidmaze-medium-navigate-v0 (GCBC)
python main.py --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-humanoidmaze-medium-navigate-v0 (GCIVL)
python main.py --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-navigate-v0 (GCIQL)
python main.py --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-navigate-v0 (QRL)
python main.py --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.001 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-navigate-v0 (CRL)
python main.py --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-navigate-v0 (HIQL)
python main.py --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/hiql.py --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100

# visual-humanoidmaze-large-navigate-v0 (GCBC)
python main.py --env_name=visual-humanoidmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-humanoidmaze-large-navigate-v0 (GCIVL)
python main.py --env_name=visual-humanoidmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-large-navigate-v0 (GCIQL)
python main.py --env_name=visual-humanoidmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-large-navigate-v0 (QRL)
python main.py --env_name=visual-humanoidmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.001 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-large-navigate-v0 (CRL)
python main.py --env_name=visual-humanoidmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-large-navigate-v0 (HIQL)
python main.py --env_name=visual-humanoidmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/hiql.py --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100

# visual-humanoidmaze-giant-navigate-v0 (GCBC)
python main.py --env_name=visual-humanoidmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-humanoidmaze-giant-navigate-v0 (GCIVL)
python main.py --env_name=visual-humanoidmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-giant-navigate-v0 (GCIQL)
python main.py --env_name=visual-humanoidmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-giant-navigate-v0 (QRL)
python main.py --env_name=visual-humanoidmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.001 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-giant-navigate-v0 (CRL)
python main.py --env_name=visual-humanoidmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-giant-navigate-v0 (HIQL)
python main.py --env_name=visual-humanoidmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100

# visual-humanoidmaze-medium-stitch-v0 (GCBC)
python main.py --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-humanoidmaze-medium-stitch-v0 (GCIVL)
python main.py --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-stitch-v0 (GCIQL)
python main.py --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-stitch-v0 (QRL)
python main.py --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.001 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-stitch-v0 (CRL)
python main.py --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-stitch-v0 (HIQL)
python main.py --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100

# visual-humanoidmaze-large-stitch-v0 (GCBC)
python main.py --env_name=visual-humanoidmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-humanoidmaze-large-stitch-v0 (GCIVL)
python main.py --env_name=visual-humanoidmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-large-stitch-v0 (GCIQL)
python main.py --env_name=visual-humanoidmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-large-stitch-v0 (QRL)
python main.py --env_name=visual-humanoidmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.001 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-large-stitch-v0 (CRL)
python main.py --env_name=visual-humanoidmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-large-stitch-v0 (HIQL)
python main.py --env_name=visual-humanoidmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100

# visual-humanoidmaze-giant-stitch-v0 (GCBC)
python main.py --env_name=visual-humanoidmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small
# visual-humanoidmaze-giant-stitch-v0 (GCIVL)
python main.py --env_name=visual-humanoidmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-giant-stitch-v0 (GCIQL)
python main.py --env_name=visual-humanoidmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-giant-stitch-v0 (QRL)
python main.py --env_name=visual-humanoidmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.001 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-giant-stitch-v0 (CRL)
python main.py --env_name=visual-humanoidmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-giant-stitch-v0 (HIQL)
python main.py --env_name=visual-humanoidmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100

# cube-single-play-v0 (GCBC)
python main.py --env_name=cube-single-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# cube-single-play-v0 (GCIVL)
python main.py --env_name=cube-single-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-single-play-v0 (GCIQL)
python main.py --env_name=cube-single-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# cube-single-play-v0 (QRL)
python main.py --env_name=cube-single-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# cube-single-play-v0 (CRL)
python main.py --env_name=cube-single-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# cube-single-play-v0 (HIQL)
python main.py --env_name=cube-single-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# cube-double-play-v0 (GCBC)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# cube-double-play-v0 (GCIVL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-double-play-v0 (GCIQL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# cube-double-play-v0 (QRL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# cube-double-play-v0 (CRL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# cube-double-play-v0 (HIQL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# cube-triple-play-v0 (GCBC)
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# cube-triple-play-v0 (GCIVL)
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-triple-play-v0 (GCIQL)
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# cube-triple-play-v0 (QRL)
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# cube-triple-play-v0 (CRL)
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# cube-triple-play-v0 (HIQL)
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# cube-quadruple-play-v0 (GCBC)
python main.py --env_name=cube-quadruple-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# cube-quadruple-play-v0 (GCIVL)
python main.py --env_name=cube-quadruple-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-quadruple-play-v0 (GCIQL)
python main.py --env_name=cube-quadruple-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# cube-quadruple-play-v0 (QRL)
python main.py --env_name=cube-quadruple-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# cube-quadruple-play-v0 (CRL)
python main.py --env_name=cube-quadruple-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# cube-quadruple-play-v0 (HIQL)
python main.py --env_name=cube-quadruple-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# cube-single-noisy-v0 (GCBC)
python main.py --env_name=cube-single-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# cube-single-noisy-v0 (GCIVL)
python main.py --env_name=cube-single-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-single-noisy-v0 (GCIQL)
python main.py --env_name=cube-single-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# cube-single-noisy-v0 (QRL)
python main.py --env_name=cube-single-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# cube-single-noisy-v0 (CRL)
python main.py --env_name=cube-single-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# cube-single-noisy-v0 (HIQL)
python main.py --env_name=cube-single-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# cube-double-noisy-v0 (GCBC)
python main.py --env_name=cube-double-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# cube-double-noisy-v0 (GCIVL)
python main.py --env_name=cube-double-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-double-noisy-v0 (GCIQL)
python main.py --env_name=cube-double-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# cube-double-noisy-v0 (QRL)
python main.py --env_name=cube-double-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# cube-double-noisy-v0 (CRL)
python main.py --env_name=cube-double-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# cube-double-noisy-v0 (HIQL)
python main.py --env_name=cube-double-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# cube-triple-noisy-v0 (GCBC)
python main.py --env_name=cube-triple-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# cube-triple-noisy-v0 (GCIVL)
python main.py --env_name=cube-triple-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-triple-noisy-v0 (GCIQL)
python main.py --env_name=cube-triple-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# cube-triple-noisy-v0 (QRL)
python main.py --env_name=cube-triple-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# cube-triple-noisy-v0 (CRL)
python main.py --env_name=cube-triple-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# cube-triple-noisy-v0 (HIQL)
python main.py --env_name=cube-triple-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# cube-quadruple-noisy-v0 (GCBC)
python main.py --env_name=cube-quadruple-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# cube-quadruple-noisy-v0 (GCIVL)
python main.py --env_name=cube-quadruple-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-quadruple-noisy-v0 (GCIQL)
python main.py --env_name=cube-quadruple-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# cube-quadruple-noisy-v0 (QRL)
python main.py --env_name=cube-quadruple-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# cube-quadruple-noisy-v0 (CRL)
python main.py --env_name=cube-quadruple-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# cube-quadruple-noisy-v0 (HIQL)
python main.py --env_name=cube-quadruple-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# scene-play-v0 (GCBC)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# scene-play-v0 (GCIVL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# scene-play-v0 (GCIQL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# scene-play-v0 (QRL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# scene-play-v0 (CRL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# scene-play-v0 (HIQL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# scene-noisy-v0 (GCBC)
python main.py --env_name=scene-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# scene-noisy-v0 (GCIVL)
python main.py --env_name=scene-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# scene-noisy-v0 (GCIQL)
python main.py --env_name=scene-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# scene-noisy-v0 (QRL)
python main.py --env_name=scene-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# scene-noisy-v0 (CRL)
python main.py --env_name=scene-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# scene-noisy-v0 (HIQL)
python main.py --env_name=scene-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# puzzle-3x3-play-v0 (GCBC)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# puzzle-3x3-play-v0 (GCIVL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-3x3-play-v0 (GCIQL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# puzzle-3x3-play-v0 (QRL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-3x3-play-v0 (CRL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-3x3-play-v0 (HIQL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# puzzle-4x4-play-v0 (GCBC)
python main.py --env_name=puzzle-4x4-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# puzzle-4x4-play-v0 (GCIVL)
python main.py --env_name=puzzle-4x4-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x4-play-v0 (GCIQL)
python main.py --env_name=puzzle-4x4-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# puzzle-4x4-play-v0 (QRL)
python main.py --env_name=puzzle-4x4-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-4x4-play-v0 (CRL)
python main.py --env_name=puzzle-4x4-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-4x4-play-v0 (HIQL)
python main.py --env_name=puzzle-4x4-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# puzzle-4x5-play-v0 (GCBC)
python main.py --env_name=puzzle-4x5-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# puzzle-4x5-play-v0 (GCIVL)
python main.py --env_name=puzzle-4x5-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x5-play-v0 (GCIQL)
python main.py --env_name=puzzle-4x5-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# puzzle-4x5-play-v0 (QRL)
python main.py --env_name=puzzle-4x5-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-4x5-play-v0 (CRL)
python main.py --env_name=puzzle-4x5-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-4x5-play-v0 (HIQL)
python main.py --env_name=puzzle-4x5-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# puzzle-4x6-play-v0 (GCBC)
python main.py --env_name=puzzle-4x6-play-v0 --eval_episodes=50 --agent=agents/gcbc.py
# puzzle-4x6-play-v0 (GCIVL)
python main.py --env_name=puzzle-4x6-play-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x6-play-v0 (GCIQL)
python main.py --env_name=puzzle-4x6-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
# puzzle-4x6-play-v0 (QRL)
python main.py --env_name=puzzle-4x6-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-4x6-play-v0 (CRL)
python main.py --env_name=puzzle-4x6-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-4x6-play-v0 (HIQL)
python main.py --env_name=puzzle-4x6-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# puzzle-3x3-noisy-v0 (GCBC)
python main.py --env_name=puzzle-3x3-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# puzzle-3x3-noisy-v0 (GCIVL)
python main.py --env_name=puzzle-3x3-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-3x3-noisy-v0 (GCIQL)
python main.py --env_name=puzzle-3x3-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# puzzle-3x3-noisy-v0 (QRL)
python main.py --env_name=puzzle-3x3-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# puzzle-3x3-noisy-v0 (CRL)
python main.py --env_name=puzzle-3x3-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# puzzle-3x3-noisy-v0 (HIQL)
python main.py --env_name=puzzle-3x3-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# puzzle-4x4-noisy-v0 (GCBC)
python main.py --env_name=puzzle-4x4-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# puzzle-4x4-noisy-v0 (GCIVL)
python main.py --env_name=puzzle-4x4-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x4-noisy-v0 (GCIQL)
python main.py --env_name=puzzle-4x4-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# puzzle-4x4-noisy-v0 (QRL)
python main.py --env_name=puzzle-4x4-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# puzzle-4x4-noisy-v0 (CRL)
python main.py --env_name=puzzle-4x4-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# puzzle-4x4-noisy-v0 (HIQL)
python main.py --env_name=puzzle-4x4-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# puzzle-4x5-noisy-v0 (GCBC)
python main.py --env_name=puzzle-4x5-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# puzzle-4x5-noisy-v0 (GCIVL)
python main.py --env_name=puzzle-4x5-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x5-noisy-v0 (GCIQL)
python main.py --env_name=puzzle-4x5-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# puzzle-4x5-noisy-v0 (QRL)
python main.py --env_name=puzzle-4x5-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# puzzle-4x5-noisy-v0 (CRL)
python main.py --env_name=puzzle-4x5-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# puzzle-4x5-noisy-v0 (HIQL)
python main.py --env_name=puzzle-4x5-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# puzzle-4x6-noisy-v0 (GCBC)
python main.py --env_name=puzzle-4x6-noisy-v0 --eval_episodes=50 --agent=agents/gcbc.py
# puzzle-4x6-noisy-v0 (GCIVL)
python main.py --env_name=puzzle-4x6-noisy-v0 --eval_episodes=50 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x6-noisy-v0 (GCIQL)
python main.py --env_name=puzzle-4x6-noisy-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.03
# puzzle-4x6-noisy-v0 (QRL)
python main.py --env_name=puzzle-4x6-noisy-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.03
# puzzle-4x6-noisy-v0 (CRL)
python main.py --env_name=puzzle-4x6-noisy-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# puzzle-4x6-noisy-v0 (HIQL)
python main.py --env_name=puzzle-4x6-noisy-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10

# visual-cube-single-play-v0 (GCBC)
python main.py --env_name=visual-cube-single-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-play-v0 (GCIVL)
python main.py --env_name=visual-cube-single-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-play-v0 (GCIQL)
python main.py --env_name=visual-cube-single-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-play-v0 (QRL)
python main.py --env_name=visual-cube-single-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-play-v0 (CRL)
python main.py --env_name=visual-cube-single-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-play-v0 (HIQL)
python main.py --env_name=visual-cube-single-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-cube-double-play-v0 (GCBC)
python main.py --env_name=visual-cube-double-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-play-v0 (GCIVL)
python main.py --env_name=visual-cube-double-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-play-v0 (GCIQL)
python main.py --env_name=visual-cube-double-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-play-v0 (QRL)
python main.py --env_name=visual-cube-double-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-play-v0 (CRL)
python main.py --env_name=visual-cube-double-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-play-v0 (HIQL)
python main.py --env_name=visual-cube-double-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-cube-triple-play-v0 (GCBC)
python main.py --env_name=visual-cube-triple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-play-v0 (GCIVL)
python main.py --env_name=visual-cube-triple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-play-v0 (GCIQL)
python main.py --env_name=visual-cube-triple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-play-v0 (QRL)
python main.py --env_name=visual-cube-triple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-play-v0 (CRL)
python main.py --env_name=visual-cube-triple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-play-v0 (HIQL)
python main.py --env_name=visual-cube-triple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-cube-quadruple-play-v0 (GCBC)
python main.py --env_name=visual-cube-quadruple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-play-v0 (GCIVL)
python main.py --env_name=visual-cube-quadruple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-play-v0 (GCIQL)
python main.py --env_name=visual-cube-quadruple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-play-v0 (QRL)
python main.py --env_name=visual-cube-quadruple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-play-v0 (CRL)
python main.py --env_name=visual-cube-quadruple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-play-v0 (HIQL)
python main.py --env_name=visual-cube-quadruple-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-cube-single-noisy-v0 (GCBC)
python main.py --env_name=visual-cube-single-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-noisy-v0 (GCIVL)
python main.py --env_name=visual-cube-single-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-noisy-v0 (GCIQL)
python main.py --env_name=visual-cube-single-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-noisy-v0 (QRL)
python main.py --env_name=visual-cube-single-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-noisy-v0 (CRL)
python main.py --env_name=visual-cube-single-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-single-noisy-v0 (HIQL)
python main.py --env_name=visual-cube-single-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-cube-double-noisy-v0 (GCBC)
python main.py --env_name=visual-cube-double-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-noisy-v0 (GCIVL)
python main.py --env_name=visual-cube-double-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-noisy-v0 (GCIQL)
python main.py --env_name=visual-cube-double-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-noisy-v0 (QRL)
python main.py --env_name=visual-cube-double-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-noisy-v0 (CRL)
python main.py --env_name=visual-cube-double-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-double-noisy-v0 (HIQL)
python main.py --env_name=visual-cube-double-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-cube-triple-noisy-v0 (GCBC)
python main.py --env_name=visual-cube-triple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-noisy-v0 (GCIVL)
python main.py --env_name=visual-cube-triple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-noisy-v0 (GCIQL)
python main.py --env_name=visual-cube-triple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-noisy-v0 (QRL)
python main.py --env_name=visual-cube-triple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-noisy-v0 (CRL)
python main.py --env_name=visual-cube-triple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-triple-noisy-v0 (HIQL)
python main.py --env_name=visual-cube-triple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-cube-quadruple-noisy-v0 (GCBC)
python main.py --env_name=visual-cube-quadruple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-noisy-v0 (GCIVL)
python main.py --env_name=visual-cube-quadruple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-noisy-v0 (GCIQL)
python main.py --env_name=visual-cube-quadruple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-noisy-v0 (QRL)
python main.py --env_name=visual-cube-quadruple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-noisy-v0 (CRL)
python main.py --env_name=visual-cube-quadruple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-cube-quadruple-noisy-v0 (HIQL)
python main.py --env_name=visual-cube-quadruple-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-scene-play-v0 (GCBC)
python main.py --env_name=visual-scene-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-play-v0 (GCIVL)
python main.py --env_name=visual-scene-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-play-v0 (GCIQL)
python main.py --env_name=visual-scene-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-play-v0 (QRL)
python main.py --env_name=visual-scene-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-play-v0 (CRL)
python main.py --env_name=visual-scene-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-play-v0 (HIQL)
python main.py --env_name=visual-scene-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-scene-noisy-v0 (GCBC)
python main.py --env_name=visual-scene-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-noisy-v0 (GCIVL)
python main.py --env_name=visual-scene-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-noisy-v0 (GCIQL)
python main.py --env_name=visual-scene-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-noisy-v0 (QRL)
python main.py --env_name=visual-scene-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-noisy-v0 (CRL)
python main.py --env_name=visual-scene-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-scene-noisy-v0 (HIQL)
python main.py --env_name=visual-scene-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-puzzle-3x3-play-v0 (GCBC)
python main.py --env_name=visual-puzzle-3x3-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-play-v0 (GCIVL)
python main.py --env_name=visual-puzzle-3x3-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-play-v0 (GCIQL)
python main.py --env_name=visual-puzzle-3x3-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-play-v0 (QRL)
python main.py --env_name=visual-puzzle-3x3-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-play-v0 (CRL)
python main.py --env_name=visual-puzzle-3x3-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-play-v0 (HIQL)
python main.py --env_name=visual-puzzle-3x3-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-puzzle-4x4-play-v0 (GCBC)
python main.py --env_name=visual-puzzle-4x4-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-play-v0 (GCIVL)
python main.py --env_name=visual-puzzle-4x4-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-play-v0 (GCIQL)
python main.py --env_name=visual-puzzle-4x4-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-play-v0 (QRL)
python main.py --env_name=visual-puzzle-4x4-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-play-v0 (CRL)
python main.py --env_name=visual-puzzle-4x4-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-play-v0 (HIQL)
python main.py --env_name=visual-puzzle-4x4-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-puzzle-4x5-play-v0 (GCBC)
python main.py --env_name=visual-puzzle-4x5-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-play-v0 (GCIVL)
python main.py --env_name=visual-puzzle-4x5-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-play-v0 (GCIQL)
python main.py --env_name=visual-puzzle-4x5-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-play-v0 (QRL)
python main.py --env_name=visual-puzzle-4x5-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-play-v0 (CRL)
python main.py --env_name=visual-puzzle-4x5-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-play-v0 (HIQL)
python main.py --env_name=visual-puzzle-4x5-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-puzzle-4x6-play-v0 (GCBC)
python main.py --env_name=visual-puzzle-4x6-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-play-v0 (GCIVL)
python main.py --env_name=visual-puzzle-4x6-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-play-v0 (GCIQL)
python main.py --env_name=visual-puzzle-4x6-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-play-v0 (QRL)
python main.py --env_name=visual-puzzle-4x6-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-play-v0 (CRL)
python main.py --env_name=visual-puzzle-4x6-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-play-v0 (HIQL)
python main.py --env_name=visual-puzzle-4x6-play-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-puzzle-3x3-noisy-v0 (GCBC)
python main.py --env_name=visual-puzzle-3x3-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-noisy-v0 (GCIVL)
python main.py --env_name=visual-puzzle-3x3-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-noisy-v0 (GCIQL)
python main.py --env_name=visual-puzzle-3x3-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-noisy-v0 (QRL)
python main.py --env_name=visual-puzzle-3x3-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-noisy-v0 (CRL)
python main.py --env_name=visual-puzzle-3x3-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-3x3-noisy-v0 (HIQL)
python main.py --env_name=visual-puzzle-3x3-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-puzzle-4x4-noisy-v0 (GCBC)
python main.py --env_name=visual-puzzle-4x4-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-noisy-v0 (GCIVL)
python main.py --env_name=visual-puzzle-4x4-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-noisy-v0 (GCIQL)
python main.py --env_name=visual-puzzle-4x4-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-noisy-v0 (QRL)
python main.py --env_name=visual-puzzle-4x4-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-noisy-v0 (CRL)
python main.py --env_name=visual-puzzle-4x4-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x4-noisy-v0 (HIQL)
python main.py --env_name=visual-puzzle-4x4-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-puzzle-4x5-noisy-v0 (GCBC)
python main.py --env_name=visual-puzzle-4x5-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-noisy-v0 (GCIVL)
python main.py --env_name=visual-puzzle-4x5-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-noisy-v0 (GCIQL)
python main.py --env_name=visual-puzzle-4x5-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-noisy-v0 (QRL)
python main.py --env_name=visual-puzzle-4x5-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-noisy-v0 (CRL)
python main.py --env_name=visual-puzzle-4x5-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x5-noisy-v0 (HIQL)
python main.py --env_name=visual-puzzle-4x5-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# visual-puzzle-4x6-noisy-v0 (GCBC)
python main.py --env_name=visual-puzzle-4x6-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-noisy-v0 (GCIVL)
python main.py --env_name=visual-puzzle-4x6-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-noisy-v0 (GCIQL)
python main.py --env_name=visual-puzzle-4x6-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-noisy-v0 (QRL)
python main.py --env_name=visual-puzzle-4x6-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.03 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-noisy-v0 (CRL)
python main.py --env_name=visual-puzzle-4x6-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/crl.py --agent.alpha=0.1 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5
# visual-puzzle-4x6-noisy-v0 (HIQL)
python main.py --env_name=visual-puzzle-4x6-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10

# powderworld-easy-play-v0 (GCBC)
python main.py --env_name=powderworld-easy-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-easy-play-v0 (GCIVL)
python main.py --env_name=powderworld-easy-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-easy-play-v0 (GCIQL)
python main.py --env_name=powderworld-easy-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-easy-play-v0 (QRL)
python main.py --env_name=powderworld-easy-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-easy-play-v0 (CRL)
python main.py --env_name=powderworld-easy-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-easy-play-v0 (HIQL)
python main.py --env_name=powderworld-easy-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10

# powderworld-medium-play-v0 (GCBC)
python main.py --env_name=powderworld-medium-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-medium-play-v0 (GCIVL)
python main.py --env_name=powderworld-medium-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-medium-play-v0 (GCIQL)
python main.py --env_name=powderworld-medium-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-medium-play-v0 (QRL)
python main.py --env_name=powderworld-medium-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-medium-play-v0 (CRL)
python main.py --env_name=powderworld-medium-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-medium-play-v0 (HIQL)
python main.py --env_name=powderworld-medium-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10

# powderworld-hard-play-v0 (GCBC)
python main.py --env_name=powderworld-hard-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gcbc.py --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-hard-play-v0 (GCIVL)
python main.py --env_name=powderworld-hard-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gcivl.py --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-hard-play-v0 (GCIQL)
python main.py --env_name=powderworld-hard-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-hard-play-v0 (QRL)
python main.py --env_name=powderworld-hard-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-hard-play-v0 (CRL)
python main.py --env_name=powderworld-hard-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/crl.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small
# powderworld-hard-play-v0 (HIQL)
python main.py --env_name=powderworld-hard-play-v0 --train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10
