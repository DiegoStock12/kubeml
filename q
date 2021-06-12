[33mcommit 4008e5203f75a651ec56b27729e70b4630518578[m[33m ([m[1;36mHEAD -> [m[1;32minference[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sat Jun 12 08:19:50 2021 +0200

    added new inference command asking for function and modified scheduler inference logic

[33mcommit 38409105e09c8183945508bca24b61e16bd03b32[m[33m ([m[1;31morigin/inference[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Jun 11 09:53:55 2021 +0200

    set inference by default

[33mcommit 75a689810c811a7f4459f65527c1e659db554263[m[33m ([m[1;31morigin/concurrency-improv[m[33m, [m[1;32mconcurrency-improv[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sun May 23 16:50:44 2021 +0200

    fixed build methods

[33mcommit 0c7f5107c64cf0c09d7fa86d27a8324380a9cebc[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sun May 23 14:13:01 2021 +0200

    improved mutex creation

[33mcommit a614b4e7107f29e63fd01de6822c4106d5537672[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sun May 23 13:44:29 2021 +0200

    added lock object to avoid nilpointer

[33mcommit e6beb8305ea8aeb787630c8953313b935c09aad1[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sun May 23 13:31:25 2021 +0200

    increased concurrency of merging

[33mcommit e1717ae5fd3cf86ff779aebfdceee8680c56c375[m[33m ([m[1;31morigin/final-exp[m[33m, [m[1;32mfinal-exp[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri May 21 16:05:58 2021 +0200

    add vgg to main scripts

[33mcommit e38f04650ead540798612098549ac4543bc804b9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri May 21 15:57:48 2021 +0200

    added functions for vgg experiments

[33mcommit b51672686e698bfa1492bc159420e944420294a9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri May 14 14:23:11 2021 +0200

    added final plots

[33mcommit 9f00eee8249d12b7edcd4e11644b8843f42d67d8[m[33m ([m[1;31morigin/master[m[33m, [m[1;31morigin/experiments[m[33m, [m[1;32mexperiments[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Apr 30 18:21:03 2021 +0200

    fixed deployment issues

[33mcommit 22dab85eb2640e9d94e7acc89323914cfa6e2324[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 27 17:39:20 2021 +0200

    added cpu env build

[33mcommit d2e6ddbffe6fe6d235667deff2883332a9d68011[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 26 11:51:20 2021 +0200

    added epoch to parameters and added multisteplr to the resnet function

[33mcommit 2a12a131dce3be62dfc853b9de5a56b1ac377b8b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Apr 23 16:12:26 2021 +0200

    added multiple connections in a pool to improve averaging performance

[33mcommit f2f1c264281f4d936fe761b15772d1d628102c29[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Apr 22 12:35:39 2021 +0200

    finished hooks for loading and reseting optimizer state

[33mcommit 163bdddbfb34729b0b64ac79211fc5ba44e31448[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Apr 22 12:21:53 2021 +0200

    fixed state changed before initialization of optimizer

[33mcommit 3cc2a74f8089dd1bb7435f5af1f90482ebf10c16[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Apr 22 12:01:55 2021 +0200

    started with new version of library

[33mcommit d858a5c36f08b6803d6b51aac964dcb6408e9d53[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Apr 22 00:11:52 2021 +0200

    moved optimizer creation after model step

[33mcommit 5bacd4be3f0c6a74ad4ddbfb032a6431457426cd[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 21 23:08:59 2021 +0200

    fixed loading optimizer state after setting the network to the proper device

[33mcommit 5ab0838341c5098b7891fa08a8d6ec4ea6d1915f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 21 22:42:30 2021 +0200

    added optimizer loading and saving hooks for more stability across epochs

[33mcommit a3fde8c66f0adb3068d497bbb13da3fd7db3cb0c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 21 18:49:59 2021 +0200

    added support for different transformations in train and validation

[33mcommit 891ad78e587afb15f8f1bc23455fd01db077b2d7[m[33m ([m[1;32mdistr-val[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 20 22:14:51 2021 +0200

    finished sparse experiments

[33mcommit 13a43516ade72245abfb5989b8490feb3edac6db[m[33m ([m[1;31morigin/distr-val[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 20 16:55:31 2021 +0200

    improved handling of responses from functions in train job

[33mcommit 216d96241eca6f9a45df7fdcda5c9fe3192e572e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 20 12:19:21 2021 +0200

    fixed error sending notification on validation

[33mcommit 757002a9b9a9cbf8702badd233129cb6b32ea8cb[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 20 12:03:09 2021 +0200

    fixed validation dataset loading train batches

[33mcommit ed543657edaccdfd97f8ddc918310539a8bda180[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 20 11:37:17 2021 +0200

    added support for validation in parallel

[33mcommit 1b2cd31c5c6a5323292c603413b20e39cceb2eb2[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 19 21:01:17 2021 +0200

    made model wait for val func before next epoch

[33mcommit ded861df284fa2d6d2bf9674ffc7506706f0f8e4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 19 19:28:17 2021 +0200

    added replication option to train script

[33mcommit 15f935e71a8655dc783f4f71b60057ee0dfdf129[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 19 19:08:15 2021 +0200

    Fixed error catching hook in serverless container

[33mcommit 704e5c3301cb4bc9a982df16a4611205185fbe09[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 19 18:04:58 2021 +0200

    added default error handler to the function container and apply method to the kubeml lib

[33mcommit 4d15dedeef36bc628e36ec151e45ee8f20c336ca[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 19 17:26:19 2021 +0200

    added prune histories and tasks in CLI

[33mcommit 798a4c35e7f778547cb6611bf15824b370d4229f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 19 16:12:33 2021 +0200

    created final plots

[33mcommit 6cc0dd1ded1a63f4e842e30a7b0dcf259e3ab023[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sun Apr 18 17:55:06 2021 +0200

    rollback to best working version

[33mcommit ac54db78c20ed399e69cdda29491f4d7c1c54acd[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sun Apr 18 17:49:45 2021 +0200

    made model wait for validation function to complete

[33mcommit d083893fe55997388dcea75651a357dd7ba83c9f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sun Apr 18 12:34:04 2021 +0200

    added logs and prune tasks functionality in the cli

[33mcommit 7dbcf528db3919e1f76306fd13516116926b3792[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sun Apr 18 11:29:17 2021 +0200

    debugging random timeout in task listing

[33mcommit 83a08cf39f1a54457c3631d41c14706a5c52800a[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sat Apr 17 12:59:12 2021 +0200

    Release 0.1.2

[33mcommit cd59e5cbc6f8dbd0d82c74b966c45b89976fe950[m[33m ([m[1;31morigin/refactor-network[m[33m, [m[1;32mrefactor-network[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sat Apr 17 12:57:20 2021 +0200

    refactored network in kubeml library

[33mcommit 81d6b9bada86c3bda410b17e5d21afe8777e03b7[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Apr 16 21:49:05 2021 +0200

    finished network class refactor

[33mcommit 7278cbfd81b08dac145fbf0c0d991e3c9dd47e52[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Apr 16 14:15:52 2021 +0200

    added standardize to tensorflow funcs

[33mcommit 08f5ae4e38f2cabf877afa8449bf4073bc004e9a[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Apr 16 12:48:53 2021 +0200

    added output options to the kubeml script

[33mcommit 83d113dd9fe174cd134ce121f6d28374db8b88a5[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Apr 15 18:08:01 2021 +0200

    merged experimenets with insights notebook

[33mcommit 0a8eb90d5c2f7b89ad6a1c316c6e7bd38be50a4d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Apr 15 14:08:37 2021 +0200

    added output folder and metrics folder parameter to tf train

[33mcommit a909585aa8f2e987b1e57e73742d14c9cfe75135[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Apr 15 11:50:27 2021 +0200

    Fixed tf experiments and added time callback

[33mcommit 51cd1e89111e00e7dd40bf8316332af6eff7de79[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 14 17:37:44 2021 +0200

    refactored notebooks

[33mcommit 7cb8e2276253c5305790124d1b31f989261bbd04[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 14 11:36:45 2021 +0200

    added dry option to visualize pending experiments

[33mcommit 461cec65e7853a47f5f245268a15bac043319341[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 14 11:28:37 2021 +0200

    fixed different order in return

[33mcommit 7de9040110caf4bdf74f041040ee0ee2d0bcc083[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 23:45:48 2021 +0200

    added capability of resuming experiments

[33mcommit ee09beb152aecf49b5f5a6e9a6e9c71b2b487545[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 23:15:59 2021 +0200

    fixed metrics saving error

[33mcommit fe95999e7185078d1c689254ab038a0d496338ad[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 20:31:57 2021 +0200

    added function to check missing experiments in folder

[33mcommit 0cfe3f54d54015f20c7338bd6bc048d935f2d3ff[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 19:30:09 2021 +0200

    added retries

[33mcommit d556c9fd0415164ddf79ca6f7035fa8fff4d5c9e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 18:57:32 2021 +0200

    added TF experiments class

[33mcommit aa0c415664b1265bcc28feb6f31a1dd513ff2746[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 15:53:49 2021 +0200

    fixed task not showing until started

[33mcommit 22a4b16a524b96596028e34153500ce726587b2d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 15:21:23 2021 +0200

    fixed experiment function code

[33mcommit c8665b766659e8d5ae479a82aafbaa82a1ef1180[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 14:14:21 2021 +0200

    added stop task command

[33mcommit 8db72a0f8b982df5280ef0a98895c2eccd3e8339[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 13:32:32 2021 +0200

    cleared outputs

[33mcommit e7dfa2e99bb07cdbc54c477405ea714fca0303c5[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 13:31:48 2021 +0200

    added exception handling to kill api

[33mcommit 3168fbaa01b52245945730eab76c16c952c1e72b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 13:17:31 2021 +0200

    added relative command location in experiments folder

[33mcommit 8025cf4f1792c52450d76c610b868383abfcfcfa[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 13 12:37:58 2021 +0200

    finished experiment api setup

[33mcommit 493bdc504a07a8893601ea502f77781767063548[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 12 21:39:39 2021 +0200

    scripts ready for tests

[33mcommit bc8570be4979d7d72ae0ae40b287944ad98d8afe[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 12 19:05:16 2021 +0200

    added platform arg to functions

[33mcommit 36b04a4b3a6e668949c82396b4b2df0002cc2485[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 12 18:57:38 2021 +0200

    updated function code to new version of kubeml

[33mcommit 5efdb04ecfab24fac59cedf1936a787829503b69[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 12 18:54:00 2021 +0200

    added capability to use multiple gpus to the code

[33mcommit 94842a7d7d567678a7ca89a8b7aa39b29ca722bd[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 12 16:56:16 2021 +0200

    fixed resnet func

[33mcommit a92cefb650a248a984ac4ce05ec4cc080136065f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 12 16:55:08 2021 +0200

    fixed resnet func

[33mcommit 6abd6ca27508142850807c42b95a597e4a2e767d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 12 13:43:11 2021 +0200

    changed scripts and added experiment sync

[33mcommit 983f541c1d176b6d1e0cf1bd7187d3eb397a0690[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 19:22:45 2021 +0200

    everything working in prod server

[33mcommit b76d28db95ff7283ea62b16a5c59c0295a59e5eb[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 17:08:04 2021 +0200

    fix installing torch twice dockerfile

[33mcommit 2cc9dda88b3e7bdc6386276c380aede16b6bf304[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 16:56:47 2021 +0200

    changed pytorch install location

[33mcommit 8d85c8fda23ccf33f331e9a295535f2aec381370[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 16:53:14 2021 +0200

    Added new base container and libs

[33mcommit b48ee7458dab060c0c426ee5a78199fb0f5ba6cb[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 14:08:04 2021 +0200

    added correct lib name

[33mcommit 3833197e988892cf6123ab5e1b039c445013472d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 14:05:08 2021 +0200

    fixed issue with lib name

[33mcommit 37f658f5858f76b914b1bc6f21b0a9a962428a6e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 14:04:03 2021 +0200

    added libpython requirement

[33mcommit 66340581e210b02ba459b3024655d637ce90669d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 13:52:48 2021 +0200

    changed torch dependencies

[33mcommit 04162d2030e42cc0a1439b6a9015f6ffc03b2075[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Apr 7 13:14:47 2021 +0200

    added nvidia env variables to environment container

[33mcommit 388ed89d5bb8116e22c9350e9c812b79f667ef1c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 6 19:53:16 2021 +0200

    fixed tf functions and added no grad to pytorch ones to save memory

[33mcommit 8a1b425d4900b8f59613364571d7efa211a2e5f8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Apr 6 13:50:19 2021 +0200

    updated to work with tf 2.2 and cuda 10.1

[33mcommit 81c775d5544f0f222ad0ad105dd0a0dfa617e467[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 5 22:56:58 2021 +0200

    added metrics endpoint for the server stats

[33mcommit cdf1a6e06b00c564ddcee1678c4875ff95e1654e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Apr 5 19:05:40 2021 +0200

    added distributed tf code

[33mcommit 14dd388114afe900860a9e06cd4e1d528b208e7f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Apr 1 15:28:29 2021 +0200

    added scripts to upload data

[33mcommit 2b5161ff600b95c981ccd112248faa21248bf940[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 31 13:45:24 2021 +0200

    updated server timeout

[33mcommit bf375cdff8598e425865c31ab329541ec883d6ba[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 30 17:42:06 2021 +0200

    fixed not resetting the model after updating periodically

[33mcommit 96d90765346f26db3ad2daf30c584e45b646045f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 30 17:26:11 2021 +0200

    fixed buf

[33mcommit b5fd7998405c36a7d97bdb2719e733a2b8bdef22[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 30 17:04:41 2021 +0200

    Added concurrent model update

[33mcommit d788cee521cf90f6aea78107650835a916d7b5c9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 30 15:29:44 2021 +0200

    updated environment request timeout

[33mcommit 9fe324df1a80f259e37b34c25ea7125bdd4a4c32[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 24 15:50:54 2021 +0100

    fixed deployment to hide services

[33mcommit 597c2549eaac5817c778e65abc6d91d2e6e235e6[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 24 14:11:12 2021 +0100

    Added type casting based on tensor type in parallelSGD

[33mcommit 2ccb6a5609c1320b6924a859493937de082bba3c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 24 13:42:06 2021 +0100

    added logic for dealing with non float and non array layer types

[33mcommit 97a43967e608bae15a8e9e729b76e094ff49c2d8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 24 12:17:58 2021 +0100

    fixed bug with loading labels from dataset

[33mcommit e204ad4453ad488fa1763027696c27c7c88e64e7[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 22 19:39:41 2021 +0100

    added function to compute tta

[33mcommit b36a3187f4313a5aac222c27d10b93609f748e2e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 17 17:10:57 2021 +0100

    deleted extra notebooks

[33mcommit 30b7b6a97484980fc624e58bf557c45e54f6d304[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 17 17:05:48 2021 +0100

    deleted extra folder

[33mcommit fc81873455b03dba324ec58c1d69eee9d151c760[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 17 17:03:55 2021 +0100

    extracted insights from experiments

[33mcommit 73f88f0e39db750efc317cd4a0b57a7de72c5f7e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 16 21:03:49 2021 +0100

    finished initial experiments

[33mcommit 286030135979a28246e31203c9b3e6c41c39ba92[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 16 17:06:44 2021 +0100

    disable pipeline to delete tensors

[33mcommit 2c351d5a65900885e6846a6e4fa85e64f96c29cc[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 16 16:52:14 2021 +0100

    added pipelining on the server side

[33mcommit b973222646a5a7b5c68b94bd69539cf3bd705214[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 16 14:08:36 2021 +0100

    fixed bug, model and functions were using different name formatting when saving

[33mcommit 46deb2d68fc226bf7e82774a254547367d9142ae[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 16 13:32:36 2021 +0100

    modified go code to treat each layer as only weights or bias and not both, makes it easier

[33mcommit a2391a364e5cd174cec557de6f8909364aa888bb[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 16 12:48:10 2021 +0100

    flatten labels at create time and count merge time

[33mcommit d22d3ff5cc35645089c915d20e361dc2ed3afac0[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 19:31:53 2021 +0100

    cleaned dir

[33mcommit 1191a8356a9a9ae436ea6aefac7432afd36480f6[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 19:31:06 2021 +0100

    added new version of kubeml library using statedict

[33mcommit c1a72edeb09ad0a45f100a3fc6dd2a28ad5ee390[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 19:27:23 2021 +0100

    added usage of state dict instead of modules

[33mcommit f309c1f8c47a9f79a5c8f4abedc16b9c8f921686[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 19:20:24 2021 +0100

    all working except for memory issues with bigger networks

[33mcommit 4eb90a06295ecd61d532015b3d903d8f5d5b2ab2[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 16:37:02 2021 +0100

    fixed service creation

[33mcommit 9f0345739918bec854bdfac0e64c454dcc87f08c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 16:13:40 2021 +0100

    updated storage dockerfile

[33mcommit f4554f4c03b8283654901565bce8dbe29a799394[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 14:33:16 2021 +0100

    added tta and max acc experiments

[33mcommit 8fb97e381584873ba7fe2e2dc5e892b050b43fff[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 13:08:47 2021 +0100

    adapted functions to new kubeml version

[33mcommit 0ba57a130af46380b361780e2ecd76ce030788ab[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 15 12:57:51 2021 +0100

    added datasets and tested experiment wrapper

[33mcommit b905b024783f804b53d66d5541df1aa709983422[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 9 13:55:11 2021 +0100

    fixed format

[33mcommit 87442a2de499f41344cfe61cb8c189ac561e1364[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Mar 5 23:22:23 2021 +0100

    fixed validation potential deadlock

[33mcommit e059cadb8d8729bd40f81b4cbef4bdbef308ccc7[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Mar 5 19:06:35 2021 +0100

    fixed issue with validation sync

[33mcommit 3be9a9c122cf079400b67c1519fb567d49aa7118[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Mar 5 18:56:09 2021 +0100

    added k=-1 in the python code

[33mcommit 6602b120b1dd8a1ca51b2da10ca9740f24c2675e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Mar 5 18:38:42 2021 +0100

    added validation goal

[33mcommit f6bb1b404391c8baea932273b3136be5aa52eb61[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Mar 5 18:12:26 2021 +0100

    added K and goal accuracy parameters to the cli

[33mcommit 9e67da1d392f78adf301c51c3cae51057a36c929[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Mar 5 18:03:43 2021 +0100

    added function to wait for service to be ready at creation time

[33mcommit e0d2b53e232028a44c330d44d4b97df6a4e282a0[m[33m ([m[1;32mk-avg[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 22:30:25 2021 +0100

    all working with K-AVG strategy

[33mcommit 2594071d34e337b053bed53a35de7bb0f8f3ce9b[m[33m ([m[1;31morigin/k-avg[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 19:14:34 2021 +0100

    fixed issues with python lib not loading appropriate minibatches

[33mcommit b41d8b610fec588ad83e5d6fd5cc5df79848d3c0[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 18:12:52 2021 +0100

    added merge sync

[33mcommit f8cb5eac3214f701621c30f7380ac963c2e55a42[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 17:40:52 2021 +0100

    fixed issues and added debugging

[33mcommit 3a3d178a50a4e106f8bc92dd71574ded8c124747[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 17:02:20 2021 +0100

    added new image and pull policy

[33mcommit 6971453532166c657e02d7e26e2e2628c905281e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 16:49:20 2021 +0100

    fixed yaml issue

[33mcommit da4b15423c2ceaaa08b3a1099a681cc6123df7f8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 16:43:56 2021 +0100

    fixed yaml issue

[33mcommit 848ddd3bf8214e2feecce71ef6cc20f0348f3502[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 16:40:38 2021 +0100

    fixed yaml issue

[33mcommit 3c441982431fa4fb47caad0a9b36c1f7591c8da5[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 16:37:17 2021 +0100

    fixed nil pointer in waitgroup

[33mcommit 9454bf2d7be4fd81b0fdc2e9226af752a9542e6b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 16:18:43 2021 +0100

    fixed retries

[33mcommit 16fa39c4e630b8f9d904988cbd3fb639f04f24e5[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 16:05:06 2021 +0100

    added image caching better

[33mcommit 54f0fa7e05625dbbbbf5e9c7540b60178d242a83[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 15:45:26 2021 +0100

    Skip deleting pod for debugging

[33mcommit e000b01a5dd272c07b4d9fa3dc4e0a303b7c9946[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Mar 4 14:36:34 2021 +0100

    added mod caching

[33mcommit 785f106769965dfc9b6d08b16457cf549a258b3b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 20:32:52 2021 +0100

    added version

[33mcommit 7ee7d9ae9745534a96a3723357b786a97aacd054[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 19:37:53 2021 +0100

    fixed issue preventing causing parsing arguments even when dataset is not needed

[33mcommit cbe3ffeccf4e825e276c5101ad9a31a7b96aab6b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 18:18:49 2021 +0100

    added correct image version

[33mcommit c8dd3d43fc3501c4ed15477a224fc7795697a417[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 18:11:45 2021 +0100

    fixed kubeml python lib

[33mcommit c88bd16c4266568d28bebb54c569114769d539f4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 17:20:45 2021 +0100

    changed image version

[33mcommit a6fd43e667ba99d0387399703d74069f9bbb8ffc[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 16:50:52 2021 +0100

    fixed typo in train api route

[33mcommit 8b4bc1ba171d4563ef3f1c547104ead4dbd85a25[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 14:16:55 2021 +0100

    Added new image with fixed jobpod version

[33mcommit 5dc1cf1f0649543fed1793b37cb6f5350b6ee0e9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 14:12:26 2021 +0100

    Added new image

[33mcommit 517ac1c2232de4531cad9e3c0b68009eea7f88d8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 14:00:50 2021 +0100

    added new kubeml version

[33mcommit cc61ef1915049e4571331c4263bdc34cd09c102d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 13:22:56 2021 +0100

    added K param

[33mcommit 29f1a0d6fe5b8b93fb8e400befb9ab88b28c8933[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Mar 3 13:03:08 2021 +0100

    added service creation

[33mcommit f5d2d0b7833419d3abdfc8dee606e1ae6e3fc21f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 23:29:04 2021 +0100

    if function exits with error quit without sending id to the merger

[33mcommit cc0ea7bf2bb10bf0bc07b4281c1ded109d7c69e0[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 23:20:31 2021 +0100

    added select clause for better syntax

[33mcommit 5260769a85a6dfe6eda0d87595018c5f16b91942[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 23:03:43 2021 +0100

    Finished merger

[33mcommit 13f35fbfeb5716466192562264f66f0f19d5f1d7[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 23:00:43 2021 +0100

    finished the merger

[33mcommit 08b51c8f1f456ad5a96d86eeabc45ebc72b24999[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 21:42:08 2021 +0100

    Fixed typo in subset calculation

[33mcommit 70f26c9c23817064dd4d2bdc1776a9da7fe56d22[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 20:53:19 2021 +0100

    still things to finish

[33mcommit c25116f2276894e832639641c7335232f6779462[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 19:25:58 2021 +0100

    created function index

[33mcommit 39fe2c5e9e3a0bbaf83674f5592e6022f1ccb3e4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 18:28:36 2021 +0100

    added ps notification in python code

[33mcommit bfb0be98e4cb73a74210e2cc22434c4b1da72635[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 16:30:23 2021 +0100

    added function extra data

[33mcommit 7adea1fe8d6443fdb46646b6510ed07c30d404ec[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 13:29:38 2021 +0100

    added tag

[33mcommit 0a87af0ebdb48f5a4759f371a7df7f7df228a0a4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 13:28:38 2021 +0100

    small fix

[33mcommit 3b9c8ece860fbde326419547716ad90d6af592eb[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 13:15:02 2021 +0100

    fixed small bug in metrics update

[33mcommit 0b0b34b38e80740e075231456829eb0488cb3f3a[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 13:09:29 2021 +0100

    added options to the train job

[33mcommit 954f3a7d60c31e82efa113bb4b3e541adb06d0b2[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Mar 2 12:36:49 2021 +0100

    Added options to the train request

[33mcommit 6e51d745cffaa50be1110bab0e42d7b906dc2291[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 1 20:40:52 2021 +0100

    added grafana dashboard

[33mcommit debdf680f7fffe425391c771067c5d7372032339[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 1 19:19:15 2021 +0100

    Fixed chart values parameters

[33mcommit 7d90d4c6e2bd75d6f390cc9b9d05ee4e5e041766[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 1 19:18:21 2021 +0100

    added deployment shell script

[33mcommit a454c3e3c87ae1ba1f48726d2f30ae953d946a7a[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 1 18:22:01 2021 +0100

    Added get controller url auto

[33mcommit 4fe54c46a24f30c89afad8357767b6de372eeddd[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 1 16:49:41 2021 +0100

    updated image version

[33mcommit c7c9479245cfec2de3c5452e2d7078eb8bde4df3[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 1 16:24:53 2021 +0100

    Updated so everything uses the kubeml namespace

[33mcommit f2c15f1beaa30251b208f999b8046e1888881c91[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Mar 1 13:37:57 2021 +0100

    Reformatted model class and fixed variable naming issues

[33mcommit 6895cd72619e4489f92f5ff1b5290af2d2110b60[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sat Feb 27 20:50:59 2021 +0100

    added environment folder

[33mcommit 54e05322e7c23505dbab9c8e0281fbe696431c6e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Sat Feb 27 17:33:05 2021 +0100

    improved result parsing

[33mcommit 6df35086eebae17b30dad40ae2cf7dd9e1fe5cd2[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Feb 25 11:50:03 2021 +0100

    Added batch

[33mcommit 5f61c5e5e686cef7d11df28a373e12664a848f59[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Feb 25 11:39:44 2021 +0100

    updated README

[33mcommit 2a573dfb17612cb1fb98fefc165f6d94341921bd[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Feb 24 19:18:05 2021 +0100

    Defined experiments helper objects and methods

[33mcommit f5f22f862ba1ebe8c56d9d7ab24f7610768e6b98[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Feb 24 18:07:13 2021 +0100

    Added task list

[33mcommit 387b9adb9be13b9cf4917f366b552cf06962e0e4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Feb 24 17:19:22 2021 +0100

    Added functions for experiments

[33mcommit 985bc07b1fad6baf74686a8df06a373a8994fbd4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Feb 24 17:17:34 2021 +0100

    Added functions for networks

[33mcommit 3b5d19c28e2733c3cc7a16c502e7634aa76efa09[m[33m ([m[1;31morigin/final-fixes[m[33m, [m[1;32mtests[m[33m, [m[1;32mfinal-fixes[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Feb 24 13:00:37 2021 +0100

    added init check for kubeml errors

[33mcommit a7412e731c8ec38fa690d404fcfa8202e50d5ce6[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 17:19:29 2021 +0100

    Added controller service search tests

[33mcommit d6f748242e18bd4d9a9427a562c30d0c1c36f5d1[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 16:56:32 2021 +0100

    Fixed environment name in chart

[33mcommit ba2c05366bea8d416522bd4a98ae1d93e7e6ebce[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 16:54:55 2021 +0100

    Added environment spec to chart

[33mcommit 13f748e8f916adfaeb17a07965fd30a7d7764056[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 16:37:10 2021 +0100

    everything working with the new errors

[33mcommit 3427becf5d659f6cef847ad695f27eab9e62359e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 13:24:18 2021 +0100

    Added checks to train task creation, better

[33mcommit fb3103310b11faa1866aa3534ba1d21feea8fd99[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 13:18:03 2021 +0100

    Added checks to train task creation

[33mcommit cbda5e602a2733132548affb556f6790d06de3a4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 11:40:41 2021 +0100

    changed the jobs to use kubeml errors

[33mcommit c3084ee556cb0c9afed0f132130438c42f23ae11[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 11:20:03 2021 +0100

    Created error class

[33mcommit 698386d900dac67aefb1e8e89f0663352c8b0e1b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 10:44:38 2021 +0100

    Added experiments folder and readme

[33mcommit 54e99da7b1e93c2481b75feb15a9896c98dc9610[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 10:36:56 2021 +0100

    made network abstract and added abstract method

[33mcommit 91c8ba658c579971babd50d0125411338e8d2c1e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 23 10:33:27 2021 +0100

    Added helm instructions to the README

[33mcommit d8e1a0f209fab9fe42c88c04dcdd01fff34ec4d9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 22 18:24:43 2021 +0100

    finished chart

[33mcommit 12781c0b7a982c812ae37f85600cea8d50735941[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 22 17:59:10 2021 +0100

    Added helm charts

[33mcommit 8c30c41760c95365e84e2660781739ad57618ba9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 22 17:00:36 2021 +0100

    Tested locally successfully

[33mcommit 493d251f13fb48373725f47697da5a4dee1e9df6[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 22 12:54:06 2021 +0100

    Fixed readme

[33mcommit 2d895a069583424b7ae5c626522b7a6d5bb3b066[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 22 12:51:17 2021 +0100

    Added readme text

[33mcommit 77fbc6c5a4484a77d524ba7513e490d3e5feff40[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 22 11:55:23 2021 +0100

    Removed done chan. All communication goes through the http api

[33mcommit bd5e08469ef02702375f6b636223e514554517cb[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 22 11:51:09 2021 +0100

    Added function error parsing to init and validation

[33mcommit c549a83714e3c376114119aa08549beb690afe1a[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 22 11:43:55 2021 +0100

    fixed controller api

[33mcommit a2cea326b696a12653ef627651f23259c2c52fed[m[33m ([m[1;31morigin/kubeml-client-new[m[33m, [m[1;32mkubeml-client-new[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Feb 18 18:17:00 2021 +0100

    changed client for kubernetes-inspired one

[33mcommit bb9e850ff87339266146fabdceb5f2b09ec86af3[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Feb 18 18:01:48 2021 +0100

    finished client skeleton

[33mcommit 1a3fde514887aa10fa47bb91d8bb635f69728226[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Feb 17 18:55:20 2021 +0100

    used struct instead of interface to mimic set

[33mcommit 2957031592bb063fb42f0cf88fdd60da538b592f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Feb 17 11:47:46 2021 +0100

    changed bson import

[33mcommit a5d3fab4c9028bcdb151216e53552f55c245339c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 16 16:24:24 2021 +0100

    added special debugging message

[33mcommit 79bf1c15d4c5e4cadcf4fa27fc2dd611d1039502[m[33m ([m[1;31morigin/decouple-ps[m[33m, [m[1;32mdecouple-ps[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 16 15:34:25 2021 +0100

    Added list dataset functionality

[33mcommit 68f406a27b3ea8e2238d948d0d5e086a4ff41ba1[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 16 14:24:57 2021 +0100

    Finished error exit message

[33mcommit b3116e8e229da253f55fb74c5c1e3f193517e0e0[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 20:23:16 2021 +0100

    check for no body done right

[33mcommit 2f51ad998252e4f775a299d7995f54e65ea8a386[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 19:56:25 2021 +0100

    need to test the exiterr feature

[33mcommit 46df5a4a8cf8cee32ee90244c77dd2b89e5e7f07[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 15:46:30 2021 +0100

    Added error sending after initialize, defer

[33mcommit 731f3c7648be8f603336e6a4cc4418fb649977cd[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 15:43:32 2021 +0100

    changed finish to include error message

[33mcommit 2eb57f6c0b0305656c2923e1b8684ddd49dbbe8c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 15:27:38 2021 +0100

    Fixed fatal error instead of clearing

[33mcommit ce4e15c7bc938b9fd75d177cf18e17cc0a1aa2ea[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 15:16:51 2021 +0100

    added API error handling to the jobs

[33mcommit 673c375fc0a0e598e059b649d73f20fa32420c84[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 13:58:57 2021 +0100

    WIP: if there is no such dataset, job does not work

[33mcommit eed881ae55057bb48f71d190949590fae26d63a7[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 13:33:20 2021 +0100

    Added limit parallelism setting

[33mcommit 9368fc87ee68c875f60b444fcd10483abc651a3c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 15 13:06:10 2021 +0100

    Fixed issues with pod creation

[33mcommit ae4027be0e1969cb887fc0752dfa55c84c0ec905[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 12 20:51:45 2021 +0100

    renamed module to kubeml

[33mcommit e4686e5fef6ea62c5fcb5904f6c64ab003e60420[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 12 19:49:16 2021 +0100

    Almost finished all the transition, need final checks

[33mcommit 898638811ba0a0758154667fc08f45e6014dbbb6[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 12 18:32:18 2021 +0100

    added job client and alternate constructor

[33mcommit 599ac77916f5c43e9303f8c592fdf08e3bbb8cc5[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 12 12:54:33 2021 +0100

    WIP: Updated PS api and client

[33mcommit a733222d92c7893cb26b43c9410b403fefa1d480[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Feb 11 15:52:17 2021 +0100

    WIP:partly finished the new ps api with finish and metrics update methods

[33mcommit 9ffdc6dc475a308f65b3706b36e452cac4686e6a[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Feb 11 13:16:12 2021 +0100

    WIP:decouple PS from train job using pods

[33mcommit 7fba5396082ae774cb1bb0410fe30ea7cecec0b9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Feb 11 12:22:07 2021 +0100

    Compiling after train task refactor

[33mcommit 54fb6e80deac1d447b14ce973dbd06c284e29474[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 9 19:46:53 2021 +0100

    Renamed variable in scheduler policy for more clarity

[33mcommit f35826a58a6445a6e8f146a49964058e5573f623[m[33m ([m[1;31morigin/sched-policy[m[33m, [m[1;32msched-policy[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 9 19:17:30 2021 +0100

    migrated to scheduler policy format

[33mcommit 1c1e05d705aaff873c7394aab7247d7e4a381474[m[33m ([m[1;32mclean-after-train[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Feb 9 18:26:57 2021 +0100

    Finished cleaning

[33mcommit f727c3c6ae259ef599382353746f6b8890e3760b[m[33m ([m[1;31morigin/clean-after-train[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 8 13:23:49 2021 +0100

    Added delete tensor functionality

[33mcommit 0c7a128719a2524700e37bc99cd58fbbe3b341cd[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 8 13:13:32 2021 +0100

    Basics of deleting tensors work

[33mcommit e44ceb3c49149b20108560c54d98743d0d022925[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 8 12:34:35 2021 +0100

    added new dependencies to solve fission dependency issue

[33mcommit 77e6e2418edfb0ac2ab44f995f51bd69d6356169[m[33m ([m[1;31morigin/cli-fission[m[33m, [m[1;32mcli-fission[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Feb 8 11:52:42 2021 +0100

    Finished function deployment automation via CLI

[33mcommit f42a6bb34959b628db54d5bb238f1fe65b6f41f4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 5 18:10:17 2021 +0100

    Basics of functions finished, still more testing needed

[33mcommit fd9b43a6a7ba743ccaf79c4f079f64f03929bb46[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 5 17:50:19 2021 +0100

    package creation succeeds

[33mcommit fa72feb957289f4084abc39e11255dc4a0783260[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 5 16:46:55 2021 +0100

    Tested trigger creation, all working

[33mcommit 480d665c5ab9b38791b7b31409b395a1dade5569[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 5 14:49:03 2021 +0100

    Finished functions to create functions

[33mcommit 2173a7d3597b4cb9c03ada15e6c3d7449516382d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Feb 5 11:19:51 2021 +0100

    Added apimachinery dependency that works with go 1.12

[33mcommit c8f02a2e8777e13c63565e85ec51947fa4cc3c60[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 17:12:46 2021 +0100

    Cleaned up

[33mcommit aa2922e4aba2b6c569bc81b749a1973fdebf4291[m[33m ([m[1;31morigin/python-module[m[33m, [m[1;32mpython-module[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 17:11:39 2021 +0100

    Added list with acc and loss functionality

[33mcommit 93bb3f2fe2214006e50310ef088373c491fceb9f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 17:03:26 2021 +0100

    Function working with kubeml

[33mcommit cf8b893d27b375a5540062cfab9e1ca1864d33f8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 15:59:05 2021 +0100

    New version of module working fine

[33mcommit 390886253f42839a87a393da38bef0659cb659e0[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 14:23:17 2021 +0100

    Corrected failure in dockerfile

[33mcommit 18df4274e68c924d243fc0c8a615845e6389eecf[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 13:59:43 2021 +0100

    Changed push settings to save on workflow frequency

[33mcommit c3e50c018a602b8825847afba9a5601793049b18[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 13:58:47 2021 +0100

    Changed environment to not timeout function workers

[33mcommit eab504e33dfa126e73e29632c04119d06de7a19e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 13:04:30 2021 +0100

    Bumped python-env version, replaced default wsgi for gunicorn and installed kubeml in the container

[33mcommit 1dab930c459c7d2c49ae33d381ec58c30a54da3e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 12:50:39 2021 +0100

    cleaned up build directory

[33mcommit 1bdd84f577086b3d208666f51715be5d689a9323[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 27 12:49:05 2021 +0100

    Fixed bug with python module reloading the job id

[33mcommit e8c586b8fd2b0fae6ecb90233dc566736a691d79[m[33m ([m[1;31morigin/cli-fix[m[33m, [m[1;32mcli-fix[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 18:58:57 2021 +0100

    Fixed bug in history command arguments

[33mcommit 5fb0a9ce2a7fbff29e7059c8c68c3c303003f5d8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 18:55:30 2021 +0100

    Extended history to add get and list methods

[33mcommit ae3c61a09f5fef1ec1f769d56b719252baed8f31[m
Author: Diego Albo Martínez <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 17:21:16 2021 +0100

    Update LICENSE

[33mcommit 1221f50d4f60fc8d05c7ca5c460948a5d98a4215[m
Author: Diego Albo Martínez <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 17:19:45 2021 +0100

    Create LICENSE

[33mcommit 78dfcfec472d0cd6e845242f2deee6635904dfab[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 17:16:57 2021 +0100

    Added license

[33mcommit 2e53644a938ba35aff10954d1c049fe1a6af301b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 16:30:38 2021 +0100

    Finished python and exceptions

[33mcommit 198fc62b3231178cb427a53b56fbd88adcb23f89[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 14:11:39 2021 +0100

    Created exceptions

[33mcommit a3c32122d625fa15d2e8c6b4536258659c25d439[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 13:45:42 2021 +0100

    Improved privacy of args members

[33mcommit 00a9606882cc1b164b8974af3c018704bb6588bb[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 13:31:59 2021 +0100

    Cleaned files

[33mcommit 1567eb55cf201f250371b1a1e406546343745ab2[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 13:30:58 2021 +0100

    Changed gitignore for cleanup

[33mcommit 8a8c4eb1e4066f2ccee94c17873cebb470c62f41[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 26 13:24:00 2021 +0100

    Wrote README for python module

[33mcommit fb05f45b8bd991f12d647c4cf3f7262413846050[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 25 21:11:35 2021 +0100

    Finished python module for now

[33mcommit 0759488c82d242df320cf88d1d385b0790fa4afd[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 25 20:55:03 2021 +0100

    WIP: for some reason loading the tensors fails

[33mcommit a6069d1dbbe26483b630ae2d8e07d55f132c9463[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 25 19:38:36 2021 +0100

    Finished basic code of the python module

[33mcommit 65a7eb535b8c32460b9a296337bcb558ae4c88ac[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 25 18:00:17 2021 +0100

    Started on basics of module

[33mcommit 0fc35602b511a02923d232784cae78701f51f189[m[33m ([m[1;31morigin/metrics[m[33m, [m[1;32mmetrics[m[33m)[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 19 18:47:43 2021 +0100

    Working after refactoring

[33mcommit b527bb352ce8f57b10ee211824b6a3155d14109b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 19 18:34:20 2021 +0100

    Refactored model class so it is cleaner to load layers

[33mcommit 2ea456df290c570b5841a1f44b1d6f61bbe0f39f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 19 18:31:48 2021 +0100

    Refactored model class so it is cleaner to load layers

[33mcommit b4781674249e62fae761597302fe2b2a1f1afeb6[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 19 18:17:26 2021 +0100

    Refactored function file

[33mcommit 013e60b1db13223a693e53e1a249229b5a173b0c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 19 17:51:06 2021 +0100

    Improved consistency in parameter names cli

[33mcommit 2733a13b6e050f70fb62dc39f39a610c79d35797[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 19 13:59:55 2021 +0100

    Fixed redis and mongo names in kubernetes and redis URI in parameter server missing redis://

[33mcommit 262a647279e5bf727229470615ce11194a4d3540[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 19 13:10:00 2021 +0100

    Fixed issue with Exception

[33mcommit 9314d59b3cb62067c523500c0e50846a421493fe[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 19 12:56:26 2021 +0100

    added debug env detection to storage service

[33mcommit a54aec30dcb34fe9e03c82679891232d5d0dc8a4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 18 20:07:34 2021 +0100

    Prometheus working

[33mcommit 3bfd55c1d82fc5021361db206cb25d0cbb462f95[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 18 17:13:31 2021 +0100

    Fixed storage dockerfile

[33mcommit 408de034c50783fe306fa0ffbf6a83a1b47fe7e3[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 18 17:08:42 2021 +0100

    Added debug env check and added deployments

[33mcommit 2376bdd3d3859cbce9cb5f7c2ad1b9d3c9761f2c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 18 15:08:15 2021 +0100

    Changed metrics and deployment settings

[33mcommit e83c1dba953edd918e1073f3b04183971b75b3a1[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 18 13:53:10 2021 +0100

    Changed metrics and updated History

[33mcommit edde78f32d0d009b31774d9a5f211872097c9e3f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 13 19:43:55 2021 +0100

    Metrics working

[33mcommit 5c4cf5065867d0ae2bf564b184891a58e2beb2ea[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Jan 13 19:22:42 2021 +0100

    Started on prometheus metrics

[33mcommit 55ec0f5f463c5aaba1782be495b2e3f1bb45a9ee[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 12 20:06:23 2021 +0100

    Finished inference basics

[33mcommit f16f257c89b53a7e8fe67be0ad08bd1a860d09b9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 12 14:09:33 2021 +0100

    Created inference function

[33mcommit 1d0b5760ad7c9b9d748eef40ad0e243bb48666df[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 12 12:25:02 2021 +0100

    Finished cli for now

[33mcommit a2862cd3ad6b901a67e99829565ef6403748c0db[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 11 23:14:37 2021 +0100

    Fixed issue with loading layers (might not have bias) and also when loading models, funcId can be >=0

[33mcommit 007edbb477260151fc0a21f8b01d37a06048a671[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 11 20:30:03 2021 +0100

    Finished CLI and started on history tool

[33mcommit 31bf770d196f876bb94e3d2366cea207f482d84c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Jan 11 13:57:28 2021 +0100

    added controller client and cli

[33mcommit 302362d69cb4fc4b23dbba1f2e4980c53a3dd004[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 5 18:44:05 2021 +0100

    Finished storage proxy

[33mcommit 482f0862364d48ce77af59284d8e334a78ef215e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 5 17:23:02 2021 +0100

    Finished service

[33mcommit 11d9cf9fa5ee7b759a263a875faddcbb0e564a13[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 5 16:51:01 2021 +0100

    Finished python storage service upload

[33mcommit 7dbca1428a8e6872be579ea4dda690e9aded7444[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Jan 5 14:36:21 2021 +0100

    started on flask api

[33mcommit 4f3e53b839b087735055afd0e5f748d37103e36f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 22 20:18:58 2020 +0100

    finished autoscaling policy

[33mcommit c50767a7f6ad5e550925659c3c8fc345598ed01d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 22 17:29:11 2020 +0100

    fixed problems with deployments

[33mcommit 7290f469256003a04d1ef0234ea2fb206b74fb04[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 22 17:14:55 2020 +0100

    Fixed structure and added kubeml deployment

[33mcommit 4c2e46f5a02e68f3c4944e84da17e3e022991403[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Dec 21 20:17:27 2020 +0100

    validation working!

[33mcommit 8b0223c94c386bb05de527a8a8c24eeeb909380d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Dec 21 16:53:21 2020 +0100

    Created client folders

[33mcommit bd40012da7780ac9275817f8e7d4d2229984fe03[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 15 17:56:55 2020 +0100

    Finished sgd and model refactor, fixed problems with tensors

[33mcommit 8179498687e112f642bf704e39662714d65c2c60[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 15 17:16:21 2020 +0100

    added optimizer support and fix model not updating layers

[33mcommit 763e4c111b4bd5e9d276406af36a188a79f238ed[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Dec 14 20:09:53 2020 +0100

    Fixed not loading proper weights

[33mcommit d7cfe8b14e81991012f7d4d3dbd9b805c747b08c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Dec 11 21:36:24 2020 +0100

    fixed small err

[33mcommit c19f3582d72f9f33cfa0eb5f9ae9571cbd6b9ef6[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Dec 11 21:31:42 2020 +0100

    created scheduler queue and cleaned types

[33mcommit 76283ec41c8db8805c4fcc566e18ddd5193f55f3[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 8 12:30:00 2020 +0100

    Fixed dockerfile

[33mcommit f57de8d982ed5c2f892f8b7832f09ee3338d263d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 8 12:23:08 2020 +0100

    Changed dependencies

[33mcommit 8a74328d79d8953313e5b0c665e7fc177b411651[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 8 12:00:14 2020 +0100

    Fixed yaml

[33mcommit 6bb88366c4534a78cab48338e1b2227c21e23a39[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 8 11:58:28 2020 +0100

    Added action and tests

[33mcommit 40574ddffccb88380dcc10ef80ad9c56440cc234[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 8 00:50:01 2020 +0100

    Go mod giving issues but compiles in windows

[33mcommit 1860b552370d3351156e2d16b987f4946ff6ada7[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Dec 7 19:30:50 2020 +0100

    Able to train for multiple epochs through the loop

[33mcommit 8ad85a49293792836b7353eff033bffbaf5e015b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Dec 7 12:11:47 2020 +0100

    added load state dict stuff

[33mcommit a6ca7fb1ee5b639472db6f8dafb9d4171042c33b[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Dec 4 20:09:36 2020 +0100

    Fixed small issue

[33mcommit 8809fbeaa4117056aeead5028f37f73d611f2bc4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Dec 4 19:37:35 2020 +0100

    Separated ps and sched

[33mcommit 02bba5c390d8d7fd94c356db34951effa7af1019[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Dec 4 13:27:46 2020 +0100

    Fixed slow redis

[33mcommit 2d530438e19b5d3fd4359a85afb146c4b3b7cb55[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 1 22:39:56 2020 +0100

    Managed to read tensors as blob, much faster

[33mcommit c21eaa0d3ecc743094c25b237d6484909f881354[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Dec 1 17:15:12 2020 +0100

    Changed class so it can work with resnets

[33mcommit 79d1e816b7ee6d366a7a0efb7abe641c3ce77fd9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Nov 30 22:12:33 2020 +0100

    It works well with 1 func but not with more

[33mcommit 77f4e0bbb0c9833e9478efbde974330ea218e7ca[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Nov 30 18:57:27 2020 +0100

    Managed connection from PS to the init function

[33mcommit 14d89fe56db91a7b53bb22a16c4dc6f7c404b8a5[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Nov 27 19:33:37 2020 +0100

    added save history and train to the ps

[33mcommit 304c6ae5be1aea325535cf7ab547f01b2e10b28c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Nov 27 17:32:12 2020 +0100

    function working with mongo

[33mcommit 032d5980832b8554cdf0290bf03e1972ff03afda[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Nov 27 15:50:34 2020 +0100

    Managed to train a function

[33mcommit 13516c83f69faf2a08a6514bd99d4cff28a4b4b4[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Nov 27 13:39:08 2020 +0100

    changed version

[33mcommit 25b5d492e7fa53fad4e38470c50639eceb80fd9f[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Nov 27 13:30:55 2020 +0100

    added mongo deployment and pymongo requirement

[33mcommit fe4dc0edddac756af1bfdba27fe9bba757393a87[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 26 19:58:21 2020 +0100

    Added dataset and connection to mongodb

[33mcommit a14b34a48617abb4eb8a011acf70555414239321[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Nov 25 20:44:39 2020 +0100

    Completed PS loop (more or less)

[33mcommit a4214836e3d45f805c13777822f600ee5705d7cf[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Nov 25 17:38:00 2020 +0100

    Added redis deployment and tested interaction

[33mcommit 669e6f2e2b066d4fe806ea62d4938a599dda64d8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Nov 25 16:20:20 2020 +0100

    Added redisai requirement

[33mcommit 7884237d61bffbabddbad7a3d06d9ef435db0d05[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Nov 24 21:27:30 2020 +0100

    Fixed typo

[33mcommit bbb5890305f1afe5bf6076d21d6004d6cf4e6e4e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Nov 24 21:15:04 2020 +0100

    Started working on NN code

[33mcommit bab47ddde8936e4b07a653d010b094de3f35caff[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Tue Nov 24 19:09:18 2020 +0100

    Added controller stuff

[33mcommit abb7e89beea6134255a3cde66a28a69807694543[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Nov 23 20:52:50 2020 +0100

    Managed to save the layers properly

[33mcommit 316e15a6e5965361655342030ee786ca5b33fb22[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Nov 23 15:37:04 2020 +0100

    Added extra code for PS api and PS

[33mcommit cae3427cc81bea07be4eeddb5391c1539465e249[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Nov 20 21:03:33 2020 +0100

    tested integration with good results

[33mcommit cc0d1c4a8bb7e6639823a7e720eb0965ac845893[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Fri Nov 20 18:45:03 2020 +0100

    Tested saving the gradients mid training

[33mcommit a22e7774f87a1c8670e76584377a900fdded3a88[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 19 19:12:31 2020 +0100

    Started working on scheduler and parameter server

[33mcommit 7fde771b8f0354518ce6125f4548f217b1641ed8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 19 13:49:03 2020 +0100

    started working on the model and the api

[33mcommit 11bbe337a91886d7d7668c5cdf6f8984805b4d03[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Nov 18 21:34:01 2020 +0100

    Managed to make gorgonia work!

[33mcommit 8b82715e70b95af97325b7144f3b3b9d6bcafbc9[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Nov 18 13:33:26 2020 +0100

    Fixed structure

[33mcommit b1e50afa764fd2557b1c37e33149235b208b29f3[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Wed Nov 18 13:29:53 2020 +0100

    Changed project structure

[33mcommit bdc198de99dbe839a0bbcf025c53ed3e39b78d67[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Mon Nov 16 12:15:47 2020 +0100

    Got torch to work inside the containers

[33mcommit 4e6e2b4c6d4579bdbf99fc49f09c4e1b44b2fc8e[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 19:52:33 2020 +0100

    Added github workflow

[33mcommit 305e08a6f040685a1d61d570c50905751c1efbb7[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 17:54:06 2020 +0100

    Managed to get the fission client

[33mcommit 7d2a9f8f6824f0e7300c3db9c1ce56d636ba6cb8[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 17:49:32 2020 +0100

    Managed to get the fission client

[33mcommit 5b33913ba45b6285ebb9ce85df6e5eb155b804cf[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 17:31:01 2020 +0100

    working with fission

[33mcommit 1eef22fd3c2fd7a0d8b0eada211b69b20bfbc17d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 16:57:16 2020 +0100

    Fixed dependencies with fission

[33mcommit 8be3fb63b64567208a83b284e7c662da8b7beb8d[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 16:03:22 2020 +0100

    fixed structure

[33mcommit 97b98e94114c3169ce6769cd537154e3715a96e2[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 15:57:23 2020 +0100

    added go dir

[33mcommit a4a0ac5cdc4d9c5e02445bac2f56c747932bcc7a[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 15:53:43 2020 +0100

    fixed structure

[33mcommit f429736d98ba6727095f6294a8cdb0682e58375c[m
Author: diegostock12 <diego.albo.martinez@gmail.com>
Date:   Thu Nov 12 15:50:01 2020 +0100

    first commit
