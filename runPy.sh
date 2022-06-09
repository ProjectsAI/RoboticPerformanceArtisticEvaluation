
listML="lr dt rf gbr"

listData="datasetScientificBackground.csv datasetArtisticBackground.csv"

for Z in $listData
    do
        for Y in $listML
            do
                for X in $(seq 0 6)
                    do
                        python mainScript.py $Z $X $Y
                    done
            done
    done

