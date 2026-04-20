# GPL-v2 Source Code Written Offer

This is the GPL v2.0 Section 3 written offer for the Swin2-MoSE sidecar
distributed by **Land Connectivity Matrix ("LCM")** as part of the
Taqneen platform. It covers every binary distribution (Docker image,
tarball, deployed service) of this sidecar.

## Source availability

In accordance with GPL v2.0 Section 3, LCM makes the complete
corresponding source code available at:

- **Public mirror (authoritative)**: this repository,
  `https://github.com/geoportal-egy/taqneen-swin2-mose-sidecar`. It
  contains LCM's FastAPI serving wrapper, Dockerfile, and any LCM
  modifications, all licensed under GPL v2.0.
- **Upstream**: when pretrained Swin2-MoSE weights are vendored
  (currently pending), the canonical upstream is
  `https://github.com/IMPLabUniPr/swin2-mose`.
- **Release tarballs**: for each tagged sidecar release (`swin2-v*`),
  the corresponding source tarball is attached to the matching GitHub
  Release at
  `https://github.com/geoportal-egy/taqneen-swin2-mose-sidecar/releases`.
- **Weights mirror**: when LCM ships a sidecar image that bundles
  upstream pretrained weights, those weights will be mirrored in this
  repo (via Git LFS or Release assets) so they remain reachable for at
  least three years independent of upstream. Until real-weight mode
  ships, no weights are distributed.

## Image traceability

Every running sidecar container records its image digest
(`sha256:<digest>`) in the `imagery_tiles.sidecar_image_digest` column
of the Taqneen database. Any output produced by a binary can be traced
back to the exact source tarball it was built from.

## Validity

This offer is **valid for at least three years from the date any
Taqneen release that includes a given Swin2-MoSE sidecar build is put
into service**.

## Contact

- Email: **`contact@lcm.com`**
- GitHub issues: file an issue at
  `https://github.com/geoportal-egy/taqneen-swin2-mose-sidecar/issues`

## Out of scope

The Taqneen platform itself (Flask backend, React frontend, database
migrations, infrastructure code, and all other components outside this
sidecar) is proprietary, closed-source, all rights reserved, and owned
by LCM. It is licensed separately and contains no GPL-v2-covered code.
This offer does not extend to the Taqneen platform.

---

**Arabic / عربي**: this offer will be published bilingually (Arabic +
English) once a professional translator reviews the text. Until then,
the English text above is the operative text.

---

Copyright (c) 2026 Land Connectivity Matrix. This offer and this
repository are licensed under GPL v2.0 (see [`LICENSE`](LICENSE)).
