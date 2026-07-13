"""Shared machinery for harmonized model results dataclasses.

MNP / MORP / MNL / MDCEV all expose the canonical estimation fields ``params``
(estimate vector), ``loglik`` (mean log-likelihood), and ``n_iter``.  Older
releases used per-model names (``b``, ``ll``, ``n_iterations``, and a stored
``ll_total``).  This module centralises the deprecation shims so every results
class gets identical behaviour instead of copy-pasting it:

* :func:`legacy_init` — an ``__init__`` body (for ``@dataclass(init=False)``
  classes) that accepts legacy construction kwargs, forwards them to the
  canonical fields with a ``DeprecationWarning``, discards a computed
  ``ll_total`` kwarg (only when the class no longer stores it as a field),
  applies field defaults, and rejects unknown kwargs.
* :func:`attach_deprecated_aliases` — installs read/write ``property``
  descriptors that forward a legacy attribute to its canonical field.  These
  are attached *after* class construction so ``@dataclass`` does not mistake
  them for fields.
* :func:`attach_ll_total_alias` — installs the computed ``ll_total`` property
  (``loglik * n_obs``) for classes that no longer store it as a field.

Messages are parametrised by class name so they read, e.g.,
``MNLResults(b=...) is deprecated``.
"""

from __future__ import annotations

import dataclasses
import warnings


def legacy_init(self, kwargs: dict, legacy_map: dict[str, str], cls_name: str) -> None:
    """Populate *self* from ``kwargs``, honouring legacy names.

    Parameters
    ----------
    self
        The (``init=False``) dataclass instance being constructed.
    kwargs : dict
        Raw keyword arguments passed to ``__init__`` (mutated in place).
    legacy_map : dict[str, str]
        Mapping of ``{legacy_kwarg: canonical_field}``.
    cls_name : str
        Class name for the warning / error messages.
    """
    # 1. Translate rename aliases with DeprecationWarning.
    for old, new in legacy_map.items():
        if old in kwargs:
            warnings.warn(
                f"{cls_name}({old}=...) is deprecated; use {new}=... instead. "
                f"This shim will be removed in pybhatlib v1.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            if new in kwargs:
                raise TypeError(
                    f"{cls_name} received both legacy {old}= and canonical "
                    f"{new}=; pass only one"
                )
            kwargs[new] = kwargs.pop(old)

    field_names = {f.name for f in dataclasses.fields(self)}

    # 2. ll_total: deprecated computed quantity — discard only when the class
    #    no longer stores it as a field (otherwise it is a normal field).
    if "ll_total" in kwargs and "ll_total" not in field_names:
        warnings.warn(
            f"{cls_name}(ll_total=...) is deprecated; ll_total is now computed "
            f"as loglik * n_obs and should not be passed explicitly. "
            f"This shim will be removed in pybhatlib v1.0.",
            DeprecationWarning,
            stacklevel=3,
        )
        kwargs.pop("ll_total")

    # 3. Assign canonical fields, applying defaults for missing optional ones.
    for f in dataclasses.fields(self):
        if f.name in kwargs:
            object.__setattr__(self, f.name, kwargs.pop(f.name))
        elif f.default is not dataclasses.MISSING:
            object.__setattr__(self, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            object.__setattr__(self, f.name, f.default_factory())  # type: ignore[misc]
        else:
            raise TypeError(f"{cls_name} missing required argument: {f.name!r}")

    # 4. Reject any remaining unknown kwargs.
    if kwargs:
        raise TypeError(
            f"{cls_name} got unexpected keyword arguments: {sorted(kwargs)}"
        )


def _make_alias(cls_name: str, old_name: str, new_name: str) -> property:
    """Return a ``property`` aliasing ``old_name`` → ``new_name`` (both warn)."""

    def _getter(self):
        warnings.warn(
            f"{cls_name}.{old_name} is deprecated; use "
            f"{cls_name}.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new_name)

    def _setter(self, value):
        warnings.warn(
            f"{cls_name}.{old_name} is deprecated; use "
            f"{cls_name}.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        setattr(self, new_name, value)

    return property(_getter, _setter)


def attach_deprecated_aliases(cls, alias_map: dict[str, str]) -> None:
    """Attach read/write deprecated-alias ``property`` descriptors to *cls*.

    Parameters
    ----------
    cls
        The results dataclass (already decorated).
    alias_map : dict[str, str]
        Mapping of ``{legacy_attr: canonical_field}``.
    """
    for old, new in alias_map.items():
        setattr(cls, old, _make_alias(cls.__name__, old, new))


def attach_ll_total_alias(cls) -> None:
    """Attach the computed ``ll_total`` property (``loglik * n_obs``) to *cls*.

    For results classes that no longer store ``ll_total`` as a field.  Reading
    returns ``loglik * n_obs``; writing back-solves ``loglik``.  Both warn.
    """
    cls_name = cls.__name__

    def _getter(self):
        warnings.warn(
            f"{cls_name}.ll_total is deprecated; use "
            f"{cls_name}.loglik (mean log-likelihood) or "
            f"``loglik * n_obs`` for the total.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.loglik * self.n_obs

    def _setter(self, value):
        warnings.warn(
            f"{cls_name}.ll_total is deprecated; use "
            f"{cls_name}.loglik (mean log-likelihood) or "
            f"``loglik * n_obs`` for the total.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.n_obs:
            self.loglik = value / self.n_obs
        else:
            self.loglik = float(value)

    cls.ll_total = property(_getter, _setter)
