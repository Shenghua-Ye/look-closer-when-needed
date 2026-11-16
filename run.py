"""Entry point for data preparation, the DEV paired runs and the protocol lock. The CAL/TEST steps of the final study
run as modules (python -m exp.exp_f1_paired, ...); scripts/run.sh calls everything in order."""
import argparse

from utils.tools import str2bool


def main():
    parser = argparse.ArgumentParser(description='Risk-controlled visual token budgets for VLMs (CS707)')

    # Basic parameters
    parser.add_argument('--task_name', type=str, default='paired', help='what to run',
                        choices=['manifest', 'plan', 'extend_plan', 'paired', 'check_decode', 'signals', 'protocol_lock',
                                 'protocol_test_sample'])
    parser.add_argument('--dataset', type=str, default='sugarcrepe', help='dataset name', choices=['sugarcrepe', 'textvqa'])
    parser.add_argument('--model', type=str, default='llava_next', help='model name (models/__init__.py)', choices=['llava_next'])
    parser.add_argument('--exp_dir', type=str, default='',
                        help='results sub-folder for a new configuration of a paired run '
                             '(default: the canonical folder, e.g. E2_sugarcrepe_paired_dev)')
    parser.add_argument('--limit', type=int, default=0, help='run only the first N questions of the frozen plan (0 = all)')
    parser.add_argument('--seed', type=int, default=0, help='library random seed (data sampling seeds are fixed in code)')

    # Device parameters
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id (ignored if CUDA_VISIBLE_DEVICES is set)')

    # Logging
    parser.add_argument('--is_logging', type=str2bool, default=True, help='tee stdout to logs/<task_name>/')

    args = parser.parse_args()

    if args.task_name == 'manifest':
        if args.dataset == 'sugarcrepe':
            from data_provider.manifest_sugarcrepe import build_manifest
        else:
            from data_provider.manifest_textvqa import build_manifest
        build_manifest()
        return

    if args.task_name in ['plan', 'extend_plan', 'paired', 'check_decode']:
        if args.dataset == 'sugarcrepe':
            from exp.exp_paired_sugarcrepe import ExpPairedSugarCrepe as Exp
        else:
            from exp.exp_paired_textvqa import ExpPairedTextVQA as Exp
        exp = Exp(args)
        if args.task_name == 'plan':
            exp.plan()
        elif args.task_name == 'extend_plan':
            exp.extend()
        elif args.task_name == 'check_decode':
            if args.dataset != 'textvqa':
                raise ValueError("check_decode applies to generation (textvqa).")
            exp.check()
        else:
            print(f'Start paired cheap/full runs on {args.dataset}...')
            exp.run()
        return

    if args.task_name == 'signals':
        if args.dataset != 'textvqa':
            raise ValueError("The confidence-signal run is defined on TextVQA (generation).")
        from exp.exp_signals import ExpSignals
        ExpSignals(args).run()
        return

    if args.task_name == 'protocol_lock':
        from exp.exp_protocol import ExpProtocol
        ExpProtocol(args).run()
        return

    if args.task_name == 'protocol_test_sample':
        from exp.exp_protocol_test import ExpProtocolTest
        ExpProtocolTest(args).run()
        return


if __name__ == '__main__':
    main()
