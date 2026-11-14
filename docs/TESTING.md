# Platform tests of the beta

Release **v0.6.0-beta**, tested by downloading the published files exactly as a user would (app zip + model parts into Downloads,
checksums verified, unzip, start, ask through the chat window's API, rate, check the saved conversation). Question: *What did Prophetess
Fanny say about angels?*

| System | Machine | Model | Ready after | Answer | Rating + conversation saved | API refuses requests without the key | No helpers left after closing |
|---|---|---|---|---|---|---|---|
| Windows Server 2025 (GitHub Actions) | 4 cores, 16 GB | Q4 | 5 s | 84 s | yes | yes | n/a (stopped by the test) |
| Ubuntu 22.04 (GitHub Actions) | 4 cores, 16 GB | Q4 | 5 s | 89 s | yes | yes | yes |
| macOS 14.8, Intel | iMac, Core i5-6500, 16 GB | Q5 | 5 s | 88 s | yes | yes | yes |
| macOS 27, Apple silicon | MacBook Pro M4 Pro, 24 GB | Q4 | 19 s | 3 s | yes | yes | yes |

The automated Windows/Ubuntu test is `.github/workflows/release-smoke-test.yml` (Actions > release smoke test > Run workflow).

Earlier builds (terminal version) were also run on a clean Ubuntu 22.04 container (found libgomp missing; now bundled) and on a 2-core,
8 GB Windows machine (bundled Visual C++ installer runs; ~2.5 min per answer). Not yet exercised: a Windows PC without the Microsoft
Visual C++ runtime (the test machines already had it), and real users' double-click flows with the browser opening (tested by hand
on the Apple-silicon Mac only).
