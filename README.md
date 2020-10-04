# Differentiable Reasoning over Long Stories 

#### Models Available
- Graph-based: `gcn` `gat` `sgcn` `agnn` `rgcn`
- Sequence-based: `graph_boe` `graph_cnn` `graph_cnnh` `graph_rnn` `graph_lstm` `graph_gru` `graph_birnn` `graph_bilstm` `graph_bigru`
`graph_intra` `graph_multihead` `graph_stack`
- Text-based: `rn` `mac` `bilstm` `bilstm_mean` `bilstm_atten`
- CTP: `ctp_s` `ctp_l` `ctp_a` `ctp_m` `ntp`


#### Datasets
- Pre-generated: https://drive.google.com/file/d/1SEq_e1IVCDDzsBIBhoUQ5pOVH5kxRoZF/view
- Code: https://github.com/facebookresearch/clutrr/


#### Usage
- `cd clutrr-baselines`
- `export COMET_API_KEY="XXX"`
- `PYTHONPATH=. python3 codes/app/main.py`

Parameters:
- `config_id`: model name
- `ds`: dataset
- `ned`: the dimension of node embedding
- `eed`: the dimension of edge embedding
- `hd`: the dimension of hidden size
- `fi`: the number of filters
- `hi`: the number of highway layers
- `he`: the number of heads
- `hop`: the number of hops 
- `ep`: the number of epochs
- `se`: the number of seed
- `mt`: the metric to be used (1 for min val loss; 2 for max val acc)

For example:
- Normal train:
 `PYTHONPATH=. python3 codes/app/main.py' '--config_id gat` `--ds data_7c5b0e70` `--ned 10` `--eed 1500` `--hd 0` `--ep 100` `--fi 0` `--he 9` `--hi 0` `--hop 2 2 2 2`

- Repeat train for analysis:
 `PYTHONPATH=. python3 codes/app/main.py` `--config_id gat` `--ds data_523348e6` `--ned 10` `--eed 1000` `--hd 0` `--ep 100` `--fi 0` `--he 8` `--hi 0` `--se 1` `--mt 1`




#### Results Analysis
Normal analysis: 
- get optimal params for each model (csv under `logs/model`)
- draw line graph for models (jpg under `plots/dataset`)
- e.g. `python codes/analysis_res.py`
`--m gat gcn graph_bilstm graph_birnn graph_bigru graph_cnn graph_cnnh graph_boe`
`--ds data_089907f8 data_db9b8f04 data_7c5b0e70 data_06b8f2a1 data_523348e6 data_d83ecc3e`

Analysis for repeat train: 
- draw confidence line graph for models (pdf under `plots/dataset/"..._re_..."`)
- get optimal hyper-parameters for models (csv under `tmp/dataset/"..._re_..."`)
- e.g. `python codes/analysis_res.py`
`--m gat gcn graph_bilstm graph_birnn graph_bigru graph_cnn graph_cnnh graph_boe`
`--ds data_089907f8 data_db9b8f04 data_7c5b0e70 data_06b8f2a1 data_523348e6 data_d83ecc3e`
`--mt 1`






#### Information Extraction
Git clone
- https://github.com/philipperemy/Stanford-OpenIE-Python
- https://github.com/mmxgn/miniepy

Put `main.py` from `codes/IE` to the corresponding position.

### Dependencies

- Pytorch, 1.5.0
- Comet.ml
- Addict
- NLTK
- Pandas
- Pyyaml
- tqdm
- allennlp, 0.9.0
- torch-geometirc, 1.4.3






### Reference

Codebase for experiments on the [CLUTRR benchmark suite](https://github.com/facebookresearch/clutrr/).

- Paper: https://arxiv.org/abs/1908.06177
- Blog: https://www.cs.mcgill.ca/~ksinha4/introducing-clutrr/


##### Citation (bibtex)
```
@article{sinha2019clutrr,
  Author = {Koustuv Sinha and Shagun Sodhani and Jin Dong and Joelle Pineau and William L. Hamilton},
  Title = {CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text},
  Year = {2019},
  journal = {Empirical Methods of Natural Language Processing (EMNLP)},
  arxiv = {1908.06177}
}
```

##### Join the CLUTRR community

* Main Repo: https://github.com/facebookresearch/clutrr
* Website: https://www.cs.mcgill.ca/~ksinha4/clutrr/

##### License
CLUTRR-Baselines is CC-BY-NC 4.0 (Attr Non-Commercial Inter.) licensed, as found in the LICENSE file.
