package fixture

type LeaseManager struct { Owner string }

func (manager *LeaseManager) RenewLease(owner string) bool {
    return manager.Owner == owner
}
