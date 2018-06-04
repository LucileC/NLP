# Notes

## ssh tunnel
`ssh lcallebe@bigfoot.apt.ri.cmu.edu -p 2002 `

## sssh copy

https://unix.stackexchange.com/questions/106480/how-to-copy-files-from-one-machine-to-another-using-ssh

`scp -P 2002 eng-fra.txt lcallebe@bigfoot.apt.ri.cmu.edu:/home/lcallebe/DEV/NLPvirtualenv/NLP/translation/data`

## Open files from distant machine on local machine

https://stackoverflow.com/questions/37458814/how-to-open-remote-files-in-sublime-text-3

https://wrgms.com/editing-files-remotely-via-ssh-on-sublimetext-3/

https://stackoverflow.com/questions/18938950/rsub-with-sublime-and-ssh-connection-refusual

https://gist.github.com/Ninjex/9582357

## Choose which GPU you want to use

With `nvidia-smi`, check which one is available, then use the command:
`CUDA_VISIBLE_DEVICES=num python mina.py`