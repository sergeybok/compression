#BASED on https://github.com/fab-jul/torchac
import os 
import torch 
from torch.utils.cpp_extension import load
import numpy as np 
arithmetic_coding_backend = load(
  name="fast_arithmetic_coding",
  sources=[os.path.join('optimized', "fast_arithmetic_coding.cpp")],
  verbose=True)


PRECISION = 16


def encode_float_cdf(cdf_float,
                     sym,
                     needs_normalization=True,
                     check_input_bounds=False):
  """Encode symbols `sym` with potentially unnormalized floating point CDF.

  Check the README for more details.

  :param cdf_float: CDF tensor, float32, on CPU. Shape (N1, ..., Nm, Lp).
  :param sym: The symbols to encode, int16, on CPU. Shape (N1, ..., Nm).
  :param needs_normalization: if True, assume `cdf_float` is un-normalized and
    needs normalization. Otherwise only convert it, without normalizing.
  :param check_input_bounds: if True, ensure inputs have valid values.
    Important: may take significant time. Only enable to check.

  :return: byte-string, encoding `sym`.
  """
  if check_input_bounds:
    if cdf_float.min() < 0:
      raise ValueError(f'cdf_float.min() == {cdf_float.min()}, should be >=0.!')
    if cdf_float.max() > 1:
      raise ValueError(f'cdf_float.max() == {cdf_float.max()}, should be <=1.!')
    Lp = cdf_float.shape[-1]
    if sym.max() >= Lp - 1:
      raise ValueError
  cdf_int = _convert_to_int_and_normalize(cdf_float, needs_normalization)
  return encode_int16_normalized_cdf(cdf_int, sym)


def decode_float_cdf(cdf_float, byte_stream, needs_normalization=True):
  """Encode symbols in `byte_stream` with potentially unnormalized float CDF.

  Check the README for more details.

  :param cdf_float: CDF tensor, float32, on CPU. Shape (N1, ..., Nm, Lp).
  :param byte_stream: byte-stream, encoding some symbols `sym`.
  :param needs_normalization: if True, assume `cdf_float` is un-normalized and
    needs normalization. Otherwise only convert it, without normalizing.

  :return: decoded `sym` of shape (N1, ..., Nm).
  """
  cdf_int = _convert_to_int_and_normalize(cdf_float, needs_normalization)
  return decode_int16_normalized_cdf(cdf_int, byte_stream)


def encode_int16_normalized_cdf(cdf_int, sym):
  """Encode symbols `sym` with a normalized integer cdf `cdf_int`.

  Check the README for more details.

  :param cdf_int: CDF tensor, int16, on CPU. Shape (N1, ..., Nm, Lp).
  :param sym: The symbols to encode, int16, on CPU. Shape (N1, ..., Nm).

  :return: byte-string, encoding `sym`
  """
  cdf_int, sym = _check_and_reshape_inputs(cdf_int, sym)
  return torchac_backend.encode_cdf(cdf_int, sym)


def decode_int16_normalized_cdf(cdf_int, byte_stream):
  """Decode symbols in `byte_stream` with a normalized integer cdf `cdf_int`.

  Check the README for more details.

  :param cdf_int: CDF tensor, int16, on CPU. Shape (N1, ..., Nm, Lp).
  :param byte_stream: byte-stream, encoding some symbols `sym`.

  :return: decoded `sym` of shape (N1, ..., Nm).
  """
  cdf_reshaped = _check_and_reshape_inputs(cdf_int)
  # Merge the m dimensions into one.
  sym = torchac_backend.decode_cdf(cdf_reshaped, byte_stream)
  return _reshape_output(cdf_int.shape, sym)


def _check_and_reshape_inputs(cdf, sym=None):
  """Check device, dtype, and shapes."""
  if cdf.is_cuda:
    raise ValueError('CDF must be on CPU')
  if sym is not None and sym.is_cuda:
    raise ValueError('Symbols must be on CPU')
  if sym is not None and sym.dtype != torch.int16:
    raise ValueError('Symbols must be int16!')
  if sym is not None:
    if len(cdf.shape) != len(sym.shape) + 1 or cdf.shape[:-1] != sym.shape:
      raise ValueError(f'Invalid shapes of cdf={cdf.shape}, sym={sym.shape}! '
                       'The first m elements of cdf.shape must be equal to '
                       'sym.shape, and cdf should only have one more dimension.')
  Lp = cdf.shape[-1]
  cdf = cdf.reshape(-1, Lp)
  if sym is None:
    return cdf
  sym = sym.reshape(-1)
  return cdf, sym


def _reshape_output(cdf_shape, sym):
  """Reshape single dimension `sym` back to the correct spatial dimensions."""
  spatial_dimensions = cdf_shape[:-1]
  if len(sym) != np.prod(spatial_dimensions):
    raise ValueError()
  return sym.reshape(*spatial_dimensions)


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
  """Convert floatingpoint CDF to integers. See README for more info.

  The idea is the following:
  When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
    cdf \in [0, 1)
  (note that 1 should not be included.)
  We now want to convert this to int16 but make sure we do not get
  the same value twice, as this would break the arithmetic coder
  (you need a strictly monotonically increasing function).
  So, if needs_normalization==True, we multiply the input CDF
  with 2**16 - (Lp - 1). This means that now,
    cdf \in [0, 2**16 - (Lp - 1)].
  Then, in a final step, we add an arange(Lp), which is just a line with
  slope one. This ensure that for sure, we will get unique, strictly
  monotonically increasing CDFs, which are \in [0, 2**16)
  """
  Lp = cdf_float.shape[-1]
  factor = torch.tensor(
    2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
  new_max_value = factor
  if needs_normalization:
    new_max_value = new_max_value - (Lp - 1)
  cdf_float = cdf_float.mul(new_max_value)
  cdf_float = cdf_float.round()
  cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
  if needs_normalization:
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
  return cdf


def pmf_to_cdf(pmf):
  cdf = pmf.cumsum(dim=-1)
  zeros = torch.zeros(cdf.size()[:-1]).unsqueeze(-1)
  cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
  cdf_with_0 = cdf_with_0.clamp(max=1.)
  return cdf_with_0

class FastArithmeticCoding():
  """ 
  Note to self, TODO
  """
  def __init__(self) -> None:
    self.reset_state()
    pass
  def reset_state(self):
    self.state = arithmetic_coding_backend.createEmptyState()
  def load_state(self, fp:str):
    self.state = arithmetic_coding_backend.load_state(fp)
  def save_state(self, fp:str):
    arithmetic_coding_backend.save_state(self.state, fp)

  def encode_token(self, token:int, probs:torch.tensor, last=False):
    token = token.short()
    cdf = pmf_to_cdf(probs)
    cdf, token = _check_and_reshape_inputs(cdf, token)
    cdf_int = _convert_to_int_and_normalize(cdf, True)
    arithmetic_coding_backend.encode_cdf(cdf_int, token, self.state, last)

  def decode_token(self, probs:torch.tensor, first_step:bool):
    """
    torch::Tensor decode_cdf(
        const torch::Tensor& cdf, /* NHWLp */
        CodecState& state,
        bool first_step)
    """
    cdf = pmf_to_cdf(probs)
    cdf_int = _convert_to_int_and_normalize(cdf, True).short()
    token = arithmetic_coding_backend.decode_cdf(cdf_int, self.state, first_step)
    return token

