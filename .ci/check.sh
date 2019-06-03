#!/usr/bin/env bash
# Patrick Wieschollek <mail@patwie.com>

RETURN=0
FILES=`find . -type f -name "*" | grep -E "\.(cc|h|cu)$"`
for FILE in $FILES; do
    echo -ne "check file ${FILE}"
    clang-format-6.0 $FILE -style=file | cmp $FILE >/dev/null
    if [ $? -ne 0 ]; then
      echo " ... failed"
      echo "[!] INCORRECT FORMATTING! $FILE" >&2
      echo $FILE
      diff -u <(cat $FILE) <(clang-format-6.0 ${FILE} -style=file)
      # diff -u < (cat ${FILE}) < (clang-format ${FILE})
      RETURN=1
    else
      echo " ... ok"
    fi
done
exit $RETURN