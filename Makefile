.PHONY: smoke

# Smoke test (no vendored reference executed):
#  - the two all-zero candidates are invalid: a correct verifier REJECTS them
#    and exits 1 (a crash, exit 0, or exit 2 is a smoke failure);
#  - the RF=4/RP=5 reduced-round witness is valid: the verifier ACCEPTS it and
#    exits 0 (exercises the optional `rf` override).
smoke:
	@python3 submission/verify_with_official.py submission/examples/bad_zerotest.json; if [ $$? -eq 1 ]; then echo "OK bad_zerotest rejected"; else echo "FAIL bad_zerotest"; exit 1; fi
	@python3 submission/verify_with_official.py submission/examples/bad_cico.json; if [ $$? -eq 1 ]; then echo "OK bad_cico rejected"; else echo "FAIL bad_cico"; exit 1; fi
	@python3 submission/verify_with_official.py submission/examples/good_zerotest_rf4_rp5.json; if [ $$? -eq 0 ]; then echo "OK good_zerotest_rf4_rp5 accepted"; else echo "FAIL good_zerotest_rf4_rp5"; exit 1; fi
