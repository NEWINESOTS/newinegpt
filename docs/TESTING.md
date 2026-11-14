# Platform tests of the beta

| System | Machine | Earlier build (terminal session) | This release (chat window, release downloads) |
|---|---|---|---|
| macOS, Apple silicon | MacBook Pro M4 Pro, 24 GB, macOS 27 | passed: checks, answers, rated session | passed: unzip in Downloads with the model parts next to it, start, answer (19 s to ready, 3 s per answer), rating, conversation memory after reload, light/dark, clean shutdown |
| macOS, Intel | iMac, Core i5-6500, 16 GB, macOS 14.8 | passed: ~100 s first answer, ~55 s after; answers identical to Apple silicon | pending |
| Linux | Ubuntu 22.04.5 (x86-64 container), 8 GB | passed: needed libgomp, now bundled; Q4 chosen automatically | pending |
| Windows | Windows Server 2025 (GitHub Actions), 2 vCPU, 8 GB | passed: all files verified, Visual C++ installer runs, checks, question (173 s), rated session, GPU build loads | pending |

All answers in these tests match across systems. Not yet exercised: a Windows PC that lacks the Microsoft Visual C++ runtime (the bundled
installer runs, but the test machine already had the runtime).
