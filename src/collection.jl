function CollectionSem(observed, imply, loss, diff)

    observed = make_onelement_array(observed)
    imply = make_onelement_array(imply)
    loss = make_onelement_array(loss)
    diff = make_onelement_array(diff)

    return CollectionSem(observed, imply, loss, diff)
end
