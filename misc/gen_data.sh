#!/usr/bin/env bash




startLine=5000
nLines=$(( 256 * 500 ))
tmpFile="./tmp/data.txt"

mypython(){
  if [[ -n $DEBUG ]]; then
    cmd=$1
    shift
    ipython3 --matplotlib tk "$cmd" -- "$@"
  else
    python3 "$@"
  fi
}

main_fn(){
  node collect-wiki-data.js "$infile" | sed -n "1,$(( startLine + nLines + 1000 ))p"  | pv > "$tmpFile"
  $mypython ./data_gen.py \
    --input "$tmpFile" \
    --output ./tmp  \
    --skip-creation \
    --testencoding \
    --update

  cat "$tmpFile" | sed -n "$startLine,$(( startLine + nLines -1 ))p" | pv | gzip > "$outfile.gz"
}


infile=$1
outfile=$2
if [[ -z $infile || -z $outfile ]]; then
  echo "Usage: ./misc/gen_data.sh <input_text_file> <output_text_file>"
  exit -1
fi

mkdir -p "$(dirname $tmpFile)"
main_fn
